# Speech for the end-to-end harness (`app.exe --e2e`), made with Windows' own text-to-speech so
# every take says exactly the same thing on every run. 16 kHz mono 16-bit, what the engine takes.
#
#   powershell -ExecutionPolicy Bypass -File scripts\make-e2e-audio.ps1
#
# Writes engine\tests\fixtures\e2e\*.wav. The harness checks what was typed against what the
# engine heard, so a different voice on another machine changes nothing but the words' spelling.

Add-Type -AssemblyName System.Speech
$out = Join-Path $PSScriptRoot '..\engine\tests\fixtures\e2e'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$takes = [ordered]@{
    'fox'      = 'The quick brown fox jumps over the lazy dog.'
    'report'   = 'Please send the report by Friday afternoon.'
    'meeting'  = 'The team meeting moves to Thursday at ten.'
    'shout'    = 'Make it all capital letters.'
    'long'     = ('Dictation should keep working however long I talk. ' +
                  'This take goes on for more than half a minute, so the engine has to split it ' +
                  'into pieces at the pauses and join them back together. ' +
                  'Every sentence has to arrive, in the right order, with nothing repeated and ' +
                  'nothing lost between the pieces. ' +
                  'The weather was cold, the coffee was hot, and the train was late again. ' +
                  'After that we walked along the river to the old bridge and back. ' +
                  'That is the end of the long take.')
}

$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.Rate = 0
foreach ($name in $takes.Keys) {
    $path = Join-Path $out "$name.wav"
    $voice.SetOutputToWaveFile($path, $format)
    $voice.Speak($takes[$name])
    $voice.SetOutputToNull()
    $seconds = ((Get-Item $path).Length - 44) / 32000
    '{0,-8} {1,6:N1} s' -f $name, $seconds
}
$voice.Dispose()
