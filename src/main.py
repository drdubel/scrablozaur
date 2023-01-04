from pprint import pprint
import re

from slowniki import pkt_za_litere, premie_slowne, premie_literowe


def planszozwracacz():
    for _ in range(15):
        yield input().split()


def wierszoformater(wiersz, litery):
    wiersz = "-" + wiersz
    byla_litera_og = False

    for i in range(1, 16):
        byla_litera = False
        pop_znak = ""
        bufor = ""
        if not byla_litera_og or (wiersz[i] == "-" and wiersz[i - 1] == "-"):
            for znak in wiersz[i:]:
                if znak.isalpha():
                    byla_litera = True
                elif znak == pop_znak and byla_litera:
                    yield (bufor.replace("-", f"[{litery}{{0,1}}]"), len(bufor))
                bufor += znak
                pop_znak = znak
        if wiersz[i] != "-":
            byla_litera_og = True
        if bufor and byla_litera:
            yield (bufor.replace("-", f"[{litery}{{0,1}}]"), len(bufor))


def slowyceniacz(plansza, slowo, start_x, start_y, pion, il_uzytych_liter):
    wartosc_slowa = 0
    mnoznik_slowa = 1
    if pion:
        for litera, y in zip(slowo, range(start_y, start_y + len(slowo))):
            if plansza[start_x][y] == "-":
                if (start_x, y) in premie_slowne:
                    mnoznik_slowa *= premie_slowne[(start_x, y)]
                wartosc_litery = pkt_za_litere[litera]
                if (start_x, y) in premie_literowe:
                    wartosc_litery *= premie_literowe[(start_x, y)]
            wartosc_slowa += wartosc_litery
    else:
        for litera, x in zip(slowo, range(start_x, start_x + len(slowo))):
            if plansza[x][start_y] == "-":
                if (start_y, x) in premie_slowne:
                    mnoznik_slowa *= premie_slowne[(start_y, x)]
                wartosc_litery = pkt_za_litere[litera]
                if (start_y, x) in premie_literowe:
                    wartosc_litery *= premie_literowe[(start_y, x)]
            wartosc_slowa += wartosc_litery
    wartosc_slowa *= mnoznik_slowa
    if il_uzytych_liter == 7:
        wartosc_slowa += 50
    return wartosc_slowa


def planszoprzejezdzacz(plansza):
    slobufor = ""
    pusty_wiersz = ["#" for _ in range(15)]
    plansza_w_dol = plansza[1:] + [pusty_wiersz]
    plansza_w_gore = [pusty_wiersz] + plansza[:14]
    for wiersz, wiersz_nad, wiersz_pod in zip(plansza, plansza_w_gore, plansza_w_dol):
        wiersz_lewo = ["#"] + wiersz[:14]
        wiersz_prawo = wiersz[1:] + ["#"]
        for litera, litera_pop, litera_po, litera_nad, litera_pod in zip(
            wiersz, wiersz_lewo, wiersz_prawo, wiersz_nad, wiersz_pod
        ):
            slobufor += litera


def main():
    slowa = [
        open("words/dwuliterowki.txt", "r").read(),
        open("words/trzyliterowki.txt", "r").read(),
        open("words/czteroliterowki.txt", "r").read(),
        open("words/piecioliterowki.txt", "r").read(),
        open("words/szescioliterowki.txt", "r").read(),
        open("words/siedmioliterowki.txt", "r").read(),
        open("words/osmioliterowki.txt", "r").read(),
        open("words/dziewiecioliterowki.txt", "r").read(),
        open("words/dziesiecioliterowki.txt", "r").read(),
        open("words/jedenastoliterowki.txt", "r").read(),
        open("words/dwunastoliterowki.txt", "r").read(),
        open("words/trzynastoliterowki.txt", "r").read(),
        open("words/czternastoliterowki.txt", "r").read(),
        open("words/pietnastoliterowki.txt", "r").read(),
    ]
    litery_gracza = input()
    plansza = list(planszozwracacz())
    for wiersz in plansza:
        wyjscie = wierszoformater("".join(wiersz), litery_gracza)
        for wyrazenie, dl_wyrazenia in wyjscie:
            re.findall(wyrazenie, slowa[dl_wyrazenia - 2])
    # planszoprzejezdzacz(plansza)


if __name__ == "__main__":
    main()
