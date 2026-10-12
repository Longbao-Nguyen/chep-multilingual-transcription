import ctypes as C
from ctypes import wintypes as W
import os
from pathlib import Path
import sys
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from local_app.close_dialog import show_close_dialog

user=C.WinDLL('user32')
user.GetWindowThreadProcessId.argtypes=[W.HWND,C.POINTER(W.DWORD)]
user.GetWindowTextW.argtypes=[W.HWND,W.LPWSTR,C.c_int]
user.PostMessageW.argtypes=[W.HWND,W.UINT,W.WPARAM,W.LPARAM]
Enum=C.WINFUNCTYPE(W.BOOL,W.HWND,W.LPARAM)

def find_dialog(pid):
    found=[]
    @Enum
    def visit(hwnd,data):
        owner=W.DWORD()
        user.GetWindowThreadProcessId(hwnd,C.byref(owner))
        title=C.create_unicode_buffer(256)
        user.GetWindowTextW(hwnd,title,256)
        if owner.value==pid and title.value=='Chep - Close application': found.append(hwnd)
        return True
    user.EnumWindows(visit,0)
    return found[0] if found else None

if __name__=='__main__':
    for button,expected in ((1002,'background'),(1001,'quit')):
        result=[]
        thread=threading.Thread(target=lambda:result.append(show_close_dialog(threading.Event())),daemon=True)
        thread.start()
        until=time.monotonic()+15
        hwnd=None
        while time.monotonic()<until and not hwnd:
            hwnd=find_dialog(os.getpid())
            time.sleep(.1)
        if not hwnd: raise RuntimeError('Native dialog did not appear')
        user.PostMessageW(hwnd,0x400+102,button,0)
        thread.join(5)
        assert result==[expected],result
        print('NATIVE_DIALOG',expected,flush=True)
