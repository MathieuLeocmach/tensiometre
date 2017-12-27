import argparse
from telnetlib import Telnet

#How did I get there?
#From the webserver of the controller, download DT3100Software_V6793_169.254.3.100.jnlp.
#Open it with text editor. 
#Lookup the .jar files that are downloaded from the webserver (be careful of the "version" keyword). 
#Use JD-GUI to decompile and examine the code.

#Telnet port 9999 is open for admin. Password is clear in the jar. It allows to get the configuration of the socket.

s_WiPORT_ENHANCED_PASSWORD = "Gu3-s."

def checkSerialSettings(iIPAddress = '169.254.3.100'):
    with Telnet(iIPAddress, 9999) as tn:
        #tn.set_debuglevel(100)
        #wait for password prompt
        #be careful! two spaces between Password and semi column
        tn.read_until(b"Password  :")
        #write the password and press enter for setup mode
        tn.write(s_WiPORT_ENHANCED_PASSWORD.encode('ascii') + b"\r\n")
        #read the whole header of the setup mode
        sAnswer = tn.read_until(b"Your choice ? ").decode('ascii')
        #exit setup mode without saving
        tn.write(b"8\r\n")
    iIndex1 = sAnswer.find('Baudrate ')
    iIndex2 = sAnswer.find(",", iIndex1)
    if iIndex1 > -1 and iIndex2 > iIndex1:
        sAnswer = sAnswer[iIndex1 + 9: iIndex2]
        s_bSerialSettingsOK = sAnswer == "921600"
        if not s_bSerialSettingsOK:
            print('WRONG serial settings Baudrate is %s instead of %d'%(sAnswer, 921600))
    else:
        print('WRONG password')
        s_bSerialSettingsOK = False
    return s_bSerialSettingsOK
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check the serial settings of a DT3100 controller.')
    parser.add_argument('ip', nargs='?', default='169.254.3.100', help='IP address of the controller.')
    args = parser.parse_args()
    checkSerialSettings(args.ip)
