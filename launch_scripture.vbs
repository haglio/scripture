' Option Explicit is load-bearing, not tidiness.  A sanitize pass renamed the
' WScript.Shell object and missed one call site, leaving `wshShell` undeclared.
' Without Option Explicit VBScript treats an undeclared name as an empty
' Variant, so the call raised "Object required" at run time -- before the first
' AppendLog -- and the icon did nothing at all: no window, no log line, no
' error.  Declared up front, that same slip is a compile error instead.
Option Explicit

Dim fso, shell, projectRoot, sessionsDir, launcherLog
Dim pythonCmd, parentDir, sharedUiDir, cmd, dryRun

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

Function FindSharedUi(startDir)
  ' Walk up for the shared_ui checkout rather than assuming it sits exactly one
  ' level above this one.  It does for the primary checkout, and did not for an
  ' agent's worktree under .claude\worktrees\, which is two levels further down
  ' -- so launching what an agent had just built died on "No module named
  ' shared_ui", and the test that would have caught it could only skip.  The
  ' primary checkout finds it on the first candidate, exactly as before.
  Dim dir, candidate
  dir = startDir
  Do While Len(dir) > 0
    candidate = fso.BuildPath(dir, "shared_ui")
    If fso.FolderExists(candidate) Then
      FindSharedUi = candidate
      Exit Function
    End If
    If fso.GetParentFolderName(dir) = dir Then Exit Do
    dir = fso.GetParentFolderName(dir)
  Loop
  ' Nothing found -- name the original candidate anyway, so the launcher log
  ' records the path that was expected instead of an empty entry.
  FindSharedUi = fso.BuildPath(startDir, "shared_ui")
End Function

Function FindPythonCommand()
  Dim condaPython, venvPython, candidates, i
  ' Prefer conda env (has torch+CUDA for CoTracker3)
  condaPython = shell.ExpandEnvironmentStrings("%USERPROFILE%") & "\miniconda3\python.exe"
  If fso.FileExists(condaPython) Then
    FindPythonCommand = Quote(condaPython)
    Exit Function
  End If

  venvPython = projectRoot & "\.venv\Scripts\python.exe"
  If fso.FileExists(venvPython) Then
    FindPythonCommand = Quote(venvPython)
    Exit Function
  End If

  candidates = Array( _
    "python", _
    "py -3" _
  )
  For i = 0 To UBound(candidates)
    If shell.Run("cmd /c where " & Split(candidates(i), " ")(0) & " >nul 2>nul", 0, True) = 0 Then
      FindPythonCommand = candidates(i)
      Exit Function
    End If
  Next
  FindPythonCommand = ""
End Function

pythonCmd = FindPythonCommand()
If pythonCmd = "" Then
  AppendLog "ERROR: Could not find python launcher"
  If dryRun Then
    WScript.Echo "ERROR: Could not find python launcher"
    WScript.Quit 1
  End If
  MsgBox "Could not find python or py launcher.", vbCritical, "Scripture"
  WScript.Quit 1
End If

' The interpreter picked above is usually conda's, which has no shared_ui
' installed and no .pth pointing at one, so PYTHONPATH is the only thing that
' makes it importable. The parent directory alone is not enough: it resolves
' "shared_ui" to the checkout as a namespace package, whose real package sits
' one level further down (shared_ui/shared_ui/), so "from shared_ui.colors
' import ..." in gui.py still fails. Naming the checkout too fixes that, and
' the parent stays for any sibling laid out flat.
parentDir = fso.GetParentFolderName(projectRoot)
sharedUiDir = FindSharedUi(parentDir)
cmd = "cmd /c cd /d " & Quote(projectRoot) & " && set PYTHONPATH=" & parentDir & ";" & sharedUiDir & "&&" & pythonCmd & " -m scripture 1>>" & Quote(launcherLog) & " 2>&1"

AppendLog "INFO: Launching with command: " & cmd
If dryRun Then
  WScript.Echo "OK: " & cmd
  WScript.Quit 0
End If

shell.Run cmd, 0, False
