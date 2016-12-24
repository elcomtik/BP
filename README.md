# Bacelor Thesis - Prediction of object movement in 2d image

autor:
=======
Roman Danko

supervisor:
===========
Peter Beňo

Content:
===========================
Aktuálne dostupné riešenia pre počítačové spracovanie obrazu nám umožňujú relatívne jednoducho detegovať a sledovať pohybujúce sa objekty. Pre mobilný robot však môže byť užitočné vedieť aj predpovedať, kde sa bude sledovaný objekt nachádzať v blízkej budúcnosti. Robotický žonglér musí vedieť, kam letí padajúca loptička, aby bol schopný ju chytiť. Robotický brankár musí byť schopný určiť, ku ktorej bránkovej tyči má vykonať zákrok. Autonómny vozík môže výrazne zefektívniť plánovanie trajektórie, pokiaľ bude nie len vedieť, ktoré prekážky sú dynamické, ale zároveň bude vedieť aj určiť, kam by sa mohli v budúcnosti presunúť. Cieľom práce je implementovať počítačový program v programovacích jazykoch Python, R a C++, ktorý bude sledovať zvolené objekty a na základe ich predchádzajúcej trajektórie bude schopný predpovedať ich možné polohy v budúcnosti. Predpokladá sa použitie knižnice OpenCV.

Úlohy:
1. Naštudujte a prehľadne spracujte problematiku identifikácie objektov v 2D obraze pomocou knižnice OpenCV. Získajte datasety - videá obsahujúce situácie z úvodu práce, ktoré budete v práci používať. Experimenty natáčajte tak, aby ste minimalizovali skreslenia.
2. Implementujte riešenia schopné detegovať v obraze loptičku na základe tvaru a farby.
3. Implementujte riešenie schopné detegovať v obraze človeka.
4. Experimentálne overte vytvorené riešenia pre detekciu zvolených objektov, experimenty vyhodnoťte a zdokumentujte. Ak zaznamenáte falošné detekcie, popíšte príčiny ich vzniku.
5. Naštudujte a prehľadne spracujte problematiku sledovania detegovaných objektov. Teoreticky spracujte metódy pre sledovanie viacerých objektov. 
6. Implementujte mechanizmus sledovania objektov v datasetoch. Výstupom algoritmu by mal byť vektor pozícii, prípadne rýchlosti sledovaného objektu v čase.
7. Naštudujte a spracujte problematiku predikcie polohy v 2D priestore. Skúste spracovať čo najvyšší počet metód a prehľadne zdokumentovať ich výhody, nevýhody a možnosti použitia. 
8. Implementujte zvolené predikčné algoritmy.
9. Vyhodnoťte chybu predikcie pozície v testovacích datasetoch a výsledky spracujte vo forme tabuliek aj grafov.
10. Technicky zdokumentujte vytvorené riešenie.

