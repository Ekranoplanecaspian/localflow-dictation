; Installer hooks for LocalFlow.
;
; Tauri's generated NSIS script removes the program's own files, and the start-at-sign-in value
; (Run\LocalFlow) unless it is told the uninstall is an update (/UPDATE). Everything below is what
; it cannot know about: keeping that value across an update all the same, and the question of
; what to do with the user's settings, history and downloaded models.
;
; The models are the reason the uninstall asks. They live outside the install directory, in
; %LOCALAPPDATA%\LocalFlow, and they are several gigabytes: leaving them behind silently is
; rude, and deleting them silently means an unusually expensive re-download for somebody who
; was only reinstalling. So the choice is put to the user, and defaults to keeping them.

; An update can run the old version's uninstaller. Installing a newer version over an older one
; shows Tauri's "Uninstall before installing" choice, already selected, and Next runs the old
; uninstaller without /UPDATE. So the sign-in entry went (Tauri's own step deletes it, and up to
; 0.2.3 a hook here did too), nothing put it back, and up to 0.2.3 the user was asked whether to
; delete their settings, history and models - in the middle of an update.
;
; From 0.2.4 the installer keeps a copy of the sign-in entry before its first page and puts it
; back after installing if the uninstaller removed it. And the uninstaller tells an update from
; an uninstall: an installer runs it in place, from the install folder (`_?=` on its command
; line), while Add or remove programs gets a copy that NSIS runs from %TEMP%.
!define LOCALFLOW_RUN_KEY "Software\Microsoft\Windows\CurrentVersion\Run"
!define LOCALFLOW_KEPT_KEY "Software\LocalFlow"
!define MUI_CUSTOMFUNCTION_GUIINIT LocalFlowKeepSignIn

Function LocalFlowKeepSignIn
  ReadRegStr $0 HKCU "${LOCALFLOW_RUN_KEY}" "LocalFlow"
  ${If} $0 == ""
    DeleteRegValue HKCU "${LOCALFLOW_KEPT_KEY}" "RunBeforeUpdate"
  ${Else}
    WriteRegStr HKCU "${LOCALFLOW_KEPT_KEY}" "RunBeforeUpdate" $0
  ${EndIf}
FunctionEnd

; Stop LocalFlow's own processes: the shell, the engine and the clean-up server, but only the
; copies that run from the install folder or from %LOCALAPPDATA%\LocalFlow (where llama-server is
; downloaded). Stopping them by name alone also stopped other people's programs: "app.exe" is
; any Tauri app's default name, and LM Studio runs a llama-server.exe of its own. The folders go
; through the environment rather than into the command, so a quote in a user name cannot break it.
!macro LOCALFLOW_STOP_OWN_PROCESSES
  System::Call 'Kernel32::SetEnvironmentVariable(t "LOCALFLOW_STOP_IN", t "$INSTDIR")i'
  System::Call 'Kernel32::SetEnvironmentVariable(t "LOCALFLOW_STOP_DATA", t "$LOCALAPPDATA\LocalFlow")i'
  nsExec::Exec `powershell.exe -NoProfile -NonInteractive -Command "Get-Process app,localflow-engine,llama-server -ErrorAction SilentlyContinue | Where-Object { $$_.Path -and ($$_.Path.StartsWith($$env:LOCALFLOW_STOP_IN + '\', 'OrdinalIgnoreCase') -or $$_.Path.StartsWith($$env:LOCALFLOW_STOP_DATA + '\', 'OrdinalIgnoreCase')) } | Stop-Process -Force"`
  Sleep 800
!macroend

!macro NSIS_HOOK_PREINSTALL
  ; A running copy holds its own executable open, and the engine holds a thousand files in
  ; localflow-engine\. Without this the install fails part-written, which is worse than either
  ; outcome it was choosing between.
  !insertmacro LOCALFLOW_STOP_OWN_PROCESSES
!macroend

!macro NSIS_HOOK_POSTINSTALL
  ; The sign-in entry an older uninstaller removed during this update goes back as it was. Only
  ; after a run with pages (a silent install never uninstalls, and never kept a copy).
  ReadRegStr $0 HKCU "${LOCALFLOW_KEPT_KEY}" "RunBeforeUpdate"
  ${If} $0 != ""
  ${AndIfNot} ${Silent}
    ReadRegStr $1 HKCU "${LOCALFLOW_RUN_KEY}" "LocalFlow"
    ${If} $1 == ""
      WriteRegStr HKCU "${LOCALFLOW_RUN_KEY}" "LocalFlow" $0
    ${EndIf}
  ${EndIf}
  DeleteRegValue HKCU "${LOCALFLOW_KEPT_KEY}" "RunBeforeUpdate"
  DeleteRegKey /ifempty HKCU "${LOCALFLOW_KEPT_KEY}"
!macroend

; Whether this uninstaller was started by an installer, as part of an update: run in place, from
; the install folder, or told so with /UPDATE. Sets $0 to 1 or 0.
!macro LOCALFLOW_IS_UPDATE
  StrCpy $0 0
  ${If} $UpdateMode = 1
  ${OrIf} $EXEDIR == $INSTDIR
    StrCpy $0 1
  ${EndIf}
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  ; The sign-in entry is not deleted here: Tauri's own uninstall step does that (or keeps it for
  ; /UPDATE), and an update started from the installer gets it back from the installer's copy.
  !insertmacro LOCALFLOW_STOP_OWN_PROCESSES
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  ; Settings, history and the downloaded models. Keeping is the default because the models are
  ; a multi-gigabyte download and reinstalling is the common reason to be here. An update never
  ; asks: the user is installing LocalFlow, not removing it.
  !insertmacro LOCALFLOW_IS_UPDATE
  ${If} $0 = 0
    MessageBox MB_YESNO|MB_ICONQUESTION|MB_DEFBUTTON2 \
      "Also delete your LocalFlow settings, dictation history and downloaded models?$\n$\nThe models are several gigabytes and would have to be downloaded again." \
      /SD IDNO IDNO keep_user_data
      RMDir /r "$APPDATA\LocalFlow"
      RMDir /r "$LOCALAPPDATA\LocalFlow"
    keep_user_data:
  ${EndIf}
!macroend
