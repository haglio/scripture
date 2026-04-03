Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectRoot = fso.GetParentFolderName(WScript.ScriptFullName)
sessionsDir = projectRoot & "\sessions"
If Not fso.FolderExists(sessionsDir) Then fso.CreateFolder(sessionsDir)
launcherLog = sessionsDir & "\scripture_launcher.log"

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

Function FindPythonCommand()
  Dim condaPython, venvPython, candidates, i
  ' Prefer conda env (has torch+CUDA for CoTracker3)
  condaPython = "C:\Users\Example\miniconda3\python.exe"
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
  MsgBox "Could not find python or py launcher.", vbCritical, "Scripture"
  WScript.Quit 1
End If

parentDir = fso.GetParentFolderName(projectRoot)
cmd = "cmd /c cd /d " & Quote(projectRoot) & " && set PYTHONPATH=" & Quote(parentDir) & "&&" & pythonCmd & " -m scripture 1>>" & Quote(launcherLog) & " 2>&1"
AppendLog "INFO: Launching with command: " & cmd
shell.Run cmd, 0, False
