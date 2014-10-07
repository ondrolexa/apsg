% Structural geology toolbox for Python
% Ondrej Lexa <lexa.ondrej@gmail.com>
% 2014

First steps with APSG toolbox
=============================

APSG poskytuje několik nových tříd pro práci se strukturními daty a
jejich vizualizací. Základním kamenem je třída pro práci s vektory
`vec3`, která je odvozena z třídy `array` knihovny numpy.
Kromě běžně dostupných metod této třídy poskytuje několik nových.
Ukážeme si je v následujících příkladech.

Stažení a instalace modulu APSG
-------------------------------

Aktuální verzi modulu APSG si můžete stáhnout z adresy:<http://is.gd/apsg_modul>.
Soubor `apsg.py` uložte do pracovního adresáře, nebo kdekoliv na `PYTHONPATH`.

Načtení modulu APSG
-------------------

Modul APSG můžeme načíst jak do vlastního jmenného prostoru tak do
aktivního, což je pro interaktivní práci výhodnejší:

~~~~{.python}
>>> from apsg import *

~~~~~~~~~~~~~

Základy práce s vektory
-----------------------

Inicializace objektu vektor je možná z libovolného iterovatelného typu,
nejčastěji seznamu.

~~~~{.python}
>>> u = Vec3([1, -2, 3])
>>> v = Vec3([-2, 1, 1])

~~~~~~~~~~~~~
