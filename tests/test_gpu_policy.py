"""Exercise the shared installer/app detector with fake Windows driver telemetry."""
import json
import os
from pathlib import Path
import subprocess
import pytest
from local_app.config import ROOT

@pytest.mark.skipif(os.name!='nt',reason='Windows shared detector')
@pytest.mark.parametrize('vendor,driver,compute,compatible,reason',[
    ('10DE','572.83','8.6',True,'compatible'),
    ('10DE','520.00','8.6',False,'too old'),
    ('10DE','580.00','12.0',False,'architecture'),
    ('8086','572.83','8.6',False,'backend'),
    ('1002','572.83','8.6',False,'backend'),
    ('0000','572.83','8.6',False,'No compatible'),
])
def test_shared_gpu_eligibility(tmp_path,vendor,driver,compute,compatible,reason):
    fake=tmp_path/'smi.ps1'
    fake.write_text(f"$global:LASTEXITCODE=0\n'0, {driver}, {compute}'\n")
    script=tmp_path/'check.ps1'
    escaped=lambda p:str(p).replace("'","''")
    script.write_text(f"""function Get-CimInstance {{ [pscustomobject]@{{Name='Test GPU';PNPDeviceID='PCI\\VEN_{vendor}'}} }}
function Get-Command {{ [pscustomobject]@{{Source='{escaped(fake)}'}} }}
& '{escaped(ROOT/'config/gpu-detect.ps1')}'
""", encoding='utf-8-sig')
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(script)],
                          capture_output=True,text=True,encoding='utf-8-sig',timeout=15)
    assert result.returncode==0,result.stderr
    data=json.loads(result.stdout)
    assert data['compatible'] is compatible
    assert reason in data['reason']
    assert data['provider']==('cuda' if compatible else None)
