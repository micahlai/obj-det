import socket
import colors
import saveData

def initialize(sd, log=False):
    global save_dir
    global _log
    save_dir = sd
    _log = log

def hostPrint(text,color=""):
    if(color == ""):
        print(f"{socket.gethostname()}:{text}")
    else:
        print(f"{color}{socket.gethostname()}:{text}{colors.bcolors.ENDC}")
    if(_log):
        saveData.writeLog(text,save_dir,before=socket.gethostname())