$folder_path=$Args[0]
Get-ChildItem -Path $folder_path -Recurse | ForEach-Object {
    if ($_.IsReadOnly) {
        $_.IsReadOnly = $false
    }
    $_.LastWriteTime = Get-Date
}
