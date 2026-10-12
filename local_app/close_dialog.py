"""Windows close choice, displayed after the last browser connection closes."""
import ctypes as C
from ctypes import wintypes as W
from pathlib import Path

QUIT = 'Quit completely\nStop local work and release memory, GPU resources and application files. Your queue and completed transcripts are saved.'
BACKGROUND = 'Run in background\nKeep transcription and downloads running, and retrieve Kaggle results. Reopen Chep from its shortcut whenever you need it.'
NOTICE = 'Kaggle notebooks already submitted can keep running online after Chep quits. Reopen Chep to retrieve their results. To stop a remote job, cancel it in the queue first.'


def show_close_dialog(cancelled):
    # Activate Common Controls v6 even in the embeddable Python distribution.
    class ACTCTX(C.Structure):
        _fields_ = [('cbSize', W.ULONG), ('dwFlags', W.DWORD), ('lpSource', W.LPCWSTR),
                    ('wProcessorArchitecture', W.WORD), ('wLangId', W.WORD),
                    ('lpAssemblyDirectory', W.LPCWSTR), ('lpResourceName', W.LPCWSTR),
                    ('lpApplicationName', W.LPCWSTR), ('hModule', W.HMODULE)]
    kernel = C.WinDLL('kernel32', use_last_error=True)
    kernel.CreateActCtxW.argtypes = [C.POINTER(ACTCTX)]
    kernel.CreateActCtxW.restype = W.HANDLE
    kernel.ActivateActCtx.argtypes = [W.HANDLE, C.POINTER(C.c_size_t)]
    kernel.ActivateActCtx.restype = W.BOOL
    kernel.DeactivateActCtx.argtypes = [W.DWORD, C.c_size_t]
    kernel.ReleaseActCtx.argtypes = [W.HANDLE]
    context = ACTCTX(cbSize=C.sizeof(ACTCTX), lpSource=str(Path(__file__).with_name('chep.manifest')))
    handle = kernel.CreateActCtxW(C.byref(context))
    if handle == C.c_void_p(-1).value:
        raise C.WinError(C.get_last_error())
    cookie = C.c_size_t()
    try:
        if not kernel.ActivateActCtx(handle, C.byref(cookie)):
            raise C.WinError(C.get_last_error())
        return _show(cancelled)
    finally:
        if cookie.value:
            kernel.DeactivateActCtx(0, cookie)
        kernel.ReleaseActCtx(handle)


def _show(cancelled):
    callback_type = C.WINFUNCTYPE(C.c_long, W.HWND, W.UINT, W.WPARAM, W.LPARAM, C.c_ssize_t)
    class Button(C.Structure):
        _pack_ = 1  # CommCtrl.h wraps TaskDialog declarations in pshpack1.h.
        _fields_ = [('id', C.c_int), ('text', W.LPCWSTR)]
    class Config(C.Structure):
        _pack_ = 1
        _fields_ = [('size', W.UINT), ('parent', W.HWND), ('instance', W.HINSTANCE),
            ('flags', C.c_int), ('common', C.c_int), ('title', W.LPCWSTR), ('icon', C.c_void_p),
            ('instruction', W.LPCWSTR), ('content', W.LPCWSTR), ('button_count', W.UINT),
            ('buttons', C.POINTER(Button)), ('default', C.c_int), ('radio_count', W.UINT),
            ('radios', C.POINTER(Button)), ('default_radio', C.c_int), ('verification', W.LPCWSTR),
            ('expanded', W.LPCWSTR), ('expand_text', W.LPCWSTR), ('collapse_text', W.LPCWSTR),
            ('footer_icon', C.c_void_p), ('footer', W.LPCWSTR), ('callback', callback_type),
            ('callback_data', C.c_ssize_t), ('width', W.UINT)]
    user = C.WinDLL('user32', use_last_error=True)
    user.PostMessageW.argtypes = [W.HWND, W.UINT, W.WPARAM, W.LPARAM]
    @callback_type
    def callback(hwnd, message, wparam, lparam, data):
        if message in (0, 4) and cancelled.is_set():
            user.PostMessageW(hwnd, 0x0400 + 102, 2, 0)  # TDM_CLICK_BUTTON / IDCANCEL
        return 0
    buttons = (Button * 2)(Button(1001, QUIT), Button(1002, BACKGROUND))
    config = Config(size=C.sizeof(Config), flags=0x0008 | 0x0010 | 0x0800,
                    title='Chep - Close application', instruction='Quit Chep or keep it running?',
                    content='The last Chep browser tab has closed.\n\n' + NOTICE,
                    button_count=2, buttons=buttons, default=1002, callback=callback, width=340)
    controls = C.WinDLL('comctl32')
    controls.TaskDialogIndirect.argtypes = [C.POINTER(Config), C.POINTER(C.c_int), C.c_void_p, C.c_void_p]
    controls.TaskDialogIndirect.restype = C.c_long
    selected = C.c_int()
    result = controls.TaskDialogIndirect(C.byref(config), C.byref(selected), None, None)
    if result < 0:
        raise OSError(f'Windows close dialog failed: 0x{result & 0xffffffff:08x}')
    return 'quit' if selected.value == 1001 else 'background'
