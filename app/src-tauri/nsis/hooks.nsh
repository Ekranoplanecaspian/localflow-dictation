; Installer hooks for LocalFlow.
;
; Tauri's generated NSIS script removes the program's own files. Everything below is what it
; cannot know about: the registry value the app writes to start itself at sign-in, and the
; question of what to do with the user's settings, history and downloaded models.
;
; The models are the reason the uninstall asks. They live outside the install directory, in
; %LOCALAPPDATA%\LocalFlow, and they are several gigabytes: leaving them behind silently is
; rude, and deleting them silently means an unusually expensive re-download for somebody who
; was only reinstalling. So the choice is put to the user, and defaults to keeping them.

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
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  !insertmacro LOCALFLOW_STOP_OWN_PROCESSES

  ; Start-at-sign-in. Left behind, Windows would try to launch a program that is no longer
  ; there on every single sign-in, and the user would have no obvious way to find out why.
  DeleteRegValue HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "LocalFlow"
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  ; Settings, history and the downloaded models. Keeping is the default because the models are
  ; a multi-gigabyte download and reinstalling is the common reason to be here.
  MessageBox MB_YESNO|MB_ICONQUESTION|MB_DEFBUTTON2 \
    "Also delete your LocalFlow settings, dictation history and downloaded models?$\n$\nThe models are several gigabytes and would have to be downloaded again." \
    /SD IDNO IDNO keep_user_data
    RMDir /r "$APPDATA\LocalFlow"
    RMDir /r "$LOCALAPPDATA\LocalFlow"
  keep_user_data:
!macroend
