' Option Explicit is load-bearing, not tidiness.  A sanitize pass renamed the
' WScript.Shell object and missed one call site, leaving `wshShell` undeclared.
' Without Option Explicit VBScript treats an undeclared name as an empty
' Variant, so the call raised "Object required" at run time -- before the first
' AppendLog -- and the icon did nothing at all: no window, no log line, no
' error.  Declared up front, that same slip is a compile error instead.
Option Explicit

Dim fso, shell, projectRoot, sessionsDir, launcherLog
Dim pythonExe, cmd, dryRun

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectRoot = fso.GetParentFolderName(WScript.ScriptFullName)
sessionsDir = projectRoot & "\sessions"
If Not fso.FolderExists(sessionsDir) Then fso.CreateFolder(sessionsDir)
launcherLog = sessionsDir & "\scripture_launcher.log"

' Dry run comes from the environment, not from an argument, so that a test runs
' this file with the *same* argument list the shortcut passes -- which is none
' at all.  Reading an argument to decide was itself a bug: VBScript's And does
' not short-circuit, so `Arguments.Count > 0 And Arguments(0) = ...` evaluated
' Arguments(0) even when there were none and died with "Subscript out of range"
' on every real launch, while the test that passed an argument stayed green.
dryRun = (shell.ExpandEnvironmentStrings("%SCRIPTURE_LAUNCHER_DRY_RUN%") = "1")

Function Quote(s)
  Quote = Chr(34) & s & Chr(34)
End Function

Sub AppendLog(msg)
  On Error Resume Next
  Dim ts
  Set ts = fso.OpenTextFile(launcherLog, 8, True)
  ts.WriteLine Now & " " & msg
  ts.Close
End Sub

' Scripture runs on the project venv and nothing else -- no conda, no PATH
' search.  This file used to prefer %USERPROFILE%\miniconda3\python.exe because
' that interpreter had a CUDA build of torch and the venv's was CPU-only, which
' cotracker_tracking.py cannot use: it pins every tensor to "cuda" with no
' fallback.  That made Scripture the only app here launching on an interpreter
' the suite never runs and the repo never declares -- and the torch it picked up
' was not even conda's, but a per-user site-packages copy shared with every
' other Python on the machine.  The venv now carries the CUDA build itself (see
' CLAUDE.md), so the interpreter that runs the tests is the interpreter that
' runs the app.
'
' Falling back to a PATH python is not a lesser launch, it is a broken one: it
' has neither torch nor shared_ui, and it would die while importing, before any
' window and before any log line.  So there is no fallback.
pythonExe = projectRoot & "\.venv\Scripts\python.exe"

' No PYTHONPATH.  The venv resolves shared_ui through the editable install's
' .pth, the working directory below resolves the top-level `content` module and
' this checkout's own `scripture` package, and that is the whole path story.
cmd = "cmd /c cd /d " & Quote(projectRoot) & " && " & Quote(pythonExe) & " -m scripture 1>>" & Quote(launcherLog) & " 2>&1"

AppendLog "INFO: Launching with command: " & cmd

' The dry run reports the command it resolved and stops, without requiring the
' venv to be there: it is a check on what this file decides, and it has to work
' in CI, where the suite runs on a plain checkout that has no .venv at all.
If dryRun Then
  WScript.Echo "OK: " & cmd
  WScript.Quit 0
End If

If Not fso.FileExists(pythonExe) Then
  AppendLog "ERROR: virtual environment missing: " & pythonExe
  MsgBox "Scripture's virtual environment is missing:" & vbCrLf & pythonExe, vbCritical, "Scripture"
  WScript.Quit 1
End If

shell.Run cmd, 0, False
