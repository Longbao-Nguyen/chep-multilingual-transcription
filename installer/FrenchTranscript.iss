#define AppVersion "0.6.1"
#include "component-sizes.iss"

[Setup]
AppId={{C33F3BDD-1475-47D3-961C-8E752110272D}
AppName=Chep
AppVersion={#AppVersion}
AppPublisher=Chep
DefaultDirName={localappdata}\Programs\Chep
DefaultGroupName=Chep
DisableProgramGroupPage=yes
DisableDirPage=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\release
OutputBaseFilename=Chep-Setup-{#AppVersion}-win-x64
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
SetupLogging=yes
UninstallDisplayName=Chep
CloseApplications=yes
RestartApplications=no
InfoBeforeFile=install-info.txt

[Files]
Source: "..\release\app\*"; DestDir: "{app}"; Excludes: "\README.md,\HUONG_DAN_SU_DUNG.md"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\.build\vc_redist.x64.exe"; Flags: dontcopy
Source: "..\config\gpu-detect.ps1"; Flags: dontcopy
Source: "..\config\gpu_runtimes.json"; Flags: dontcopy

[InstallDelete]
Type: files; Name: "{app}\README.md"
Type: files; Name: "{app}\HUONG_DAN_SU_DUNG.md"
Type: filesandordirs; Name: "{app}\runtime-conversion"
Type: files; Name: "{app}\conversion-dependencies.json"
Type: files; Name: "{app}\local_app\convert_model.py"
Type: files; Name: "{app}\local_app\__pycache__\convert_model.*.pyc"
Type: filesandordirs; Name: "{app}\runtime\Lib\site-packages\nvidia"
Type: filesandordirs; Name: "{app}\runtime\Lib\site-packages\nvidia_*.dist-info"
Type: files; Name: "{app}\runtime\Lib\site-packages\ctranslate2\cudnn64_9.dll"

[Icons]
Name: "{autodesktop}\Chep"; Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\launch.py"" --data-dir ""{code:GetDataDir}"""; WorkingDir: "{app}"
Name: "{group}\Chep"; Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\launch.py"" --data-dir ""{code:GetDataDir}"""; WorkingDir: "{app}"
Name: "{group}\Uninstall Chep"; Filename: "{uninstallexe}"
Name: "{group}\Stop Chep"; Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\launch.py"" --data-dir ""{code:GetDataDir}"" --stop"; WorkingDir: "{app}"

[Run]
Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\launch.py"" --data-dir ""{code:GetDataDir}"""; WorkingDir: "{app}"; Description: "Open Chep"; Flags: nowait postinstall skipifsilent

[Registry]
Root: HKCU; Subkey: "Software\FrenchTranscriptLocal"; ValueType: string; ValueName: "DataDir"; ValueData: "{code:GetDataDir}"; Flags: uninsdeletevalue

[Code]
var
  DataPage: TInputDirWizardPage;
  OptionalPage: TInputOptionWizardPage;
  DownloadTotal: TNewStaticText;
  GPUBytes: Extended;

procedure UpdateDownloadTotal(Sender: TObject);
var Total: Extended;
begin
  Total := 0;
  if OptionalPage.Values[0] then Total := Total + GPUBytes;
  if OptionalPage.Values[1] then Total := Total + {#Size_fr};
  if OptionalPage.Values[2] then Total := Total + {#Size_en};
  if OptionalPage.Values[3] then Total := Total + {#Size_es};
  if OptionalPage.Values[4] then Total := Total + {#Size_de};
  if OptionalPage.Values[5] then Total := Total + {#Size_vi};
  DownloadTotal.Caption := 'Total download after first launch: ' + IntToStr(Round(Total / 1048576)) + ' MB.' + #13#10 + 'Downloads run in the app and can be paused or retried. Nothing is downloaded by Setup.';
end;

function GetDataDir(Param: String): String;
begin
  Result := DataPage.Values[0];
end;

procedure InitializeWizard;
var Saved, GPUInfo, Reason: String; ExitCode: Integer; GPUCompatible: Boolean;
begin
  OptionalPage := CreateInputOptionPage(wpSelectDir, 'Optional Components', 'Choose what to download after installation',
    'Leave everything unchecked to start with the CPU core. You can add components later in the app.', False, False);
  ExtractTemporaryFile('gpu-detect.ps1');
  ExtractTemporaryFile('gpu_runtimes.json');
  GPUInfo := ExpandConstant('{tmp}\gpu-check.ini');
  GPUCompatible := False;
  if Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + ExpandConstant('{tmp}\gpu-detect.ps1') + '" -IniPath "' + GPUInfo + '"', '', SW_HIDE, ewWaitUntilTerminated, ExitCode) then
    GPUCompatible := (ExitCode = 0) and (GetIniString('GPU', 'compatible', '0', GPUInfo) = '1');
  GPUBytes := StrToInt64Def(GetIniString('GPU','download_bytes','0',GPUInfo),0);
  OptionalPage.Add('Enable GPU Acceleration (' + IntToStr(Round(GPUBytes / 1048576)) + ' MB download)');
  OptionalPage.CheckListBox.ItemEnabled[0] := GPUCompatible;
  OptionalPage.Add('{#Label_fr}');
  OptionalPage.Add('{#Label_en}');
  OptionalPage.Add('{#Label_es}');
  OptionalPage.Add('{#Label_de}');
  OptionalPage.Add('{#Label_vi}');
  if not GPUCompatible then begin
    Reason := GetIniString('GPU','reason','GPU compatibility could not be checked. You can check again in the app.',GPUInfo);
    OptionalPage.SubCaptionLabel.Caption := Reason;
  end;
  OptionalPage.CheckListBox.Height := ScaleY(132);
  DownloadTotal := TNewStaticText.Create(WizardForm);
  DownloadTotal.Parent := OptionalPage.Surface;
  DownloadTotal.Top := OptionalPage.CheckListBox.Top + OptionalPage.CheckListBox.Height + ScaleY(8);
  DownloadTotal.Width := OptionalPage.SurfaceWidth;
  DownloadTotal.Height := ScaleY(48);
  DownloadTotal.AutoSize := False;
  DownloadTotal.WordWrap := True;
  OptionalPage.CheckListBox.OnClickCheck := @UpdateDownloadTotal;
  UpdateDownloadTotal(nil);
  DataPage := CreateInputDirPage(OptionalPage.ID, 'Choose your data folder',
    'Where should models and transcripts be saved?',
    'Choose a folder with enough space for your selected downloads and recordings. Your data will be kept when the application is updated or uninstalled.', False, '');
  DataPage.Add('Data folder:');
  if not RegQueryStringValue(HKCU, 'Software\FrenchTranscriptLocal', 'DataDir', Saved) then
    Saved := ExpandConstant('{localappdata}\FrenchTranscriptLocal\data');
  DataPage.Values[0] := ExpandConstant('{param:DATADIR|' + Saved + '}');
end;

procedure CurStepChanged(CurStep: TSetupStep);
var Choices, Payload: String;
begin
  if CurStep = ssPostInstall then begin
    Choices := '';
    if OptionalPage.Values[0] then Choices := Choices + '"gpu:auto",';
    if OptionalPage.Values[1] then Choices := Choices + '"model:fr",';
    if OptionalPage.Values[2] then Choices := Choices + '"model:en",';
    if OptionalPage.Values[3] then Choices := Choices + '"model:es",';
    if OptionalPage.Values[4] then Choices := Choices + '"model:de",';
    if OptionalPage.Values[5] then Choices := Choices + '"model:vi",';
    if Length(Choices) > 0 then Delete(Choices, Length(Choices), 1);
    Payload := '{"schema_version":1,"setup_id":"' + GetDateTimeString('yyyymmddhhnnsszzz', '-', ':') + '","components":[' + Choices + ']}';
    ForceDirectories(GetDataDir(''));
    if not SaveStringToFile(AddBackslash(GetDataDir('')) + 'initial_setup.json', Payload, False) then
      MsgBox('Optional choices could not be saved. You can select downloads in the app.', mbInformation, MB_OK);
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var AppDir, DataDir: String;
begin
  Result := True;
  if CurPageID = DataPage.ID then begin
    AppDir := AddBackslash(ExpandFileName(WizardDirValue));
    DataDir := AddBackslash(ExpandFileName(DataPage.Values[0]));
    if (DataPage.Values[0] = '') or (CompareText(Copy(DataDir, 1, Length(AppDir)), AppDir) = 0) then begin
      MsgBox('Choose a data folder outside the application folder so updates preserve your files.', mbError, MB_OK);
      Result := False;
    end;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var Installed, Minor: Cardinal; Code: Integer;
begin
  Result := '';
  if RegQueryDWordValue(HKLM64, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) and
     RegQueryDWordValue(HKLM64, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Minor', Minor) and
     (Installed = 1) and (Minor >= 40) then Exit;
  ExtractTemporaryFile('vc_redist.x64.exe');
  if not ShellExec('runas', ExpandConstant('{tmp}\vc_redist.x64.exe'), '/install /quiet /norestart', '', SW_HIDE, ewWaitUntilTerminated, Code) then
    Result := 'Microsoft Visual C++ runtime is required. Please allow its installation and try again.'
  else if Code = 3010 then NeedsRestart := True
  else if (Code <> 0) and (Code <> 1638) then
    Result := 'Microsoft runtime installation failed. Exit code: ' + IntToStr(Code);
end;
