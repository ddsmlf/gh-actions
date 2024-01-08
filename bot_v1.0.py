"""

    BOT WINNI THE POOH'S HOME RUN DERBY

"""

"""
-------------------------------------- Fonctions --------------------------------------
"""

from pynput.mouse import Button, Controller
import time
import pyautogui
from playsound import playsound

"""
-------------------------------------- Variables --------------------------------------
"""

color = (255, 255, 255)
NoClick = True
mouse = Controller()
DureeJeux = 0
ball = 0
centrer = (487,580)

# Valeurs pour le stage 1
temps = 2 # Duré entre apparition de la balle et clique
TotalBall = 10
demarrer = (195,406)

"""
----------------------------------------- Test -----------------------------------------
"""

"""
            TEST DE LOCALISATION
time.sleep(5)
print(mouse.position)
time.sleep(5)
print(mouse.position)
time.sleep(50)
"""

# mouse.position = (0, 0)

# time.sleep(1)
# mouse.move(1920,1080)
# print(mouse.position)
# mouse.click(Button.left, 1)

# time.sleep(1)
# mouse.press(Button.left)
# time.sleep(1)
# mouse.release(Button.left)

# mouse.position = (380, 370)
# mouse.press(Button.left)
# time.sleep(1)
# mouse.release(Button.left)

"""
------------------------------------------ Main ---------------------------------------------
"""

# Démarrer le jeux
playsound('bip.mp3')
mouse.position = demarrer
mouse.click(Button.left, 1)
mouse.position = centrer
time.sleep(4)

# Cliquer au bon moment
while ball < TotalBall :
    NoClick = True
    zone = pyautogui.screenshot(region=(0, 0, 484, 231)) #entre x,y et x,y
    if zone.getpixel((483, 230)) == color :
        time.sleep(temps)
        mouse.press(Button.left)
        time.sleep(1)
        print("trouvé")
        mouse.release(Button.left)
        time.sleep(1)
        ball += 1
        NoClick = False
            

