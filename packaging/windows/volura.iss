; Windows-installatie van Volura (Inno Setup 6).
;
;   python packaging/build.py      bouwt dist\Volura en roept ISCC aan met de versie:
;   ISCC /DAppVersion=0.1.0 packaging\windows\volura.iss
;
; Standaard installeert Volura voor de gebruiker zelf (geen beheerdersrechten nodig),
; zodat de automatische update stil kan draaien. Een update voert dezelfde installatie
; uit met /SILENT; daarna start Volura vanzelf weer (het [Run]-item zonder skipifsilent).

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "Volura"
#define AppExe "Volura.exe"
#define Root "..\.."
#define Repo "https://github.com/Lexvdpoel/Volura"

[Setup]
AppId={{244C4CBE-5970-47BE-9726-9CC6F712B186}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=L. van der Poel
AppPublisherURL=https://volura.dev
AppSupportURL={#Repo}/issues
AppUpdatesURL={#Repo}/releases
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#Root}\dist
OutputBaseFilename=Volura-windows-setup
SetupIconFile={#Root}\assets\volura.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
LicenseFile={#Root}\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
ChangesAssociations=yes
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}

[Languages]
Name: "dutch"; MessagesFile: "compiler:Languages\Dutch.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "assoc"; Description: "Open .volura files with Volura"; GroupDescription: "{cm:AssocingFileExtension,{#AppName},.volura}"

[InstallDelete]
; een update vervangt de hele programmamap, zodat er geen oude bibliotheken achterblijven
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#Root}\dist\Volura\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.volura"; ValueType: string; ValueName: ""; ValueData: "Volura.Model"; Flags: uninsdeletevalue; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\Volura.Model"; ValueType: string; ValueName: ""; ValueData: "Volura-model"; Flags: uninsdeletekey; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\Volura.Model\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#AppExe},0"; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\Volura.Model\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: assoc

[Run]
; ook na een stille update: Volura start weer
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall
