# WB-NLP: NLP w naukach społecznych
Semestr Wiosenny 2021/22

Celem grupy warsztatowej jest zapoznanie studentów z technikami przetwarzania języka naturalnego (*NLP*), a w szczególności modelowania tematów, w naukach społecznych (na przykładzie konkretnych problemów nauk politycznych). Podczas laboratorium studenci poznają wybrane techniki nienadzorowane używane w NLP oraz przykładowe zastosowania ich w naukach społecznych.

## Tematy laboratoriów

| Temat zajęć | PD/KM | Punkty |
| --- | --- | --- |
| 1. NLP i nauki społeczne, wprowadzenie w tematy (przegląd przykładowe prac i danych). |  |  |
| 2. Omówienie tematów projektów.  |  |  |
| 3. Preprocessing języka. Eksploracja danych tekstowych.  | Start PD1. |  |
| 4. Pozyskiwanie danych. Web scrapping, ekstrakcja treści artykułów, RSS.  |  |  |
| 5. Omówienie PD1. | PD1 oddanie.  | 9 |
| 6. Modelowanie tematów (topic modeling). |  |  |
| 7. Omówienie KM1 projektu. | KM1 oddanie | 9 |
| 8. Zanurzenia słów (words embeddings). Określanie optymalnej liczby tematów. |  |  |
| 9. Omówienie KM2 projektu. | KM2 Oddanie | 9 |
| 10. Structural topic modeling / Dynamic topic modeling.  |  |  |
| 11. Omówienie KM3 projektu | KM3 Oddanie | 9 |
| 12. BERT i zanurzenia zdań. Zastosowanie w modelowaniu tematów. |  |  |
| 13. Omówienie KM4 | KM4 Oddanie | 9 |
| 14. Dodatkowe konsultacje projektów |  |  |
| 15. Podsumowanie projektu |  |  |

## Prace domowe
Zadania domowe to w większości kamienie milowe związane z projektem. Wszystkie są wykonywane w grupach. Oddawanie nastęuje na zajęciach.

1. Preprocessing przykładowych danych, eksploracja.
2. Zebranie danych + eksploracja (KM1 projektu)
3. Zamodelowanie temtów, wybranie optymalnej liczby tematów. Wstępna analiza. (KM2 projektu)
4. Analiza tematów. Określenie hipotez + propozycje dodatkowych zmiennych (KM3 projektu)
5. Zamodelowanie z dodatkowymi zmiennymi, analiza (KM4 projektu)

# Projekty

## Tematy projektów
1. Dyfuzja polityk publicznych: wpływ artykułów naukowych. [3]
2. Dyfuzja polityk publicznych: wpływ big techów (na podstawie notek prasowych ogłaszających nowe technologie). [3]
3. Dyfuzja tematów w debacie publicznej: wpływ artykułów naukowych na tematy podejmowane w debacie publicznej. [3]
4. Dyfuzja tematów badawczych XAI: wpływ polityk publicznych. [3]
5. Analiza poriorytetów aktorów politycznych w EU na podstawie notek prasowych oraz ich wpływu na kształtowanie się polityk AI w EU. [1][2]

### Wysokopoziomowe kroki projektu

- zebranie danych (wskażę źródła)
- eksloracja zbioru + preprocesing i czyszczenie danych
- wstępny topic modeling,  określenie optymalnej liczby tematów, analiza tematów
- Sformuowanie hipotezy, i zebranie dodatkowych zmiennych którymi chcemy objaśniać zmienność tematów
- Zamodelowanie tematów z dodatkowymi zmiennymi, potwierdzenie lub obalenie hipotezy
- Opisanie wyników

# Punktacja (100 pkt)

- 48 praca podczas projektu
    - 9 za każdy milestone / prace domową (9*5=45 overall)
    - 3 za aktywność
- 32 raport końcowy.
- 16 prezentacja końcowa.
- 4 stosowanie dobrych praktyk wykorzystania GitHub.

# Literatura


[1]
J. Grimmer, “A Bayesian Hierarchical Topic Model for Political Texts: Measuring Expressed Agendas in Senate Press Releases,” Political Analysis, vol. 18, no. 1, pp. 1–35, ed 2010, doi: 10.1093/pan/mpp034.

[2]
N. Egami, C. J. Fong, J. Grimmer, M. E. Roberts, and B. M. Stewart, “How to Make Causal Inferences Using Texts,” *arXiv:1802.02163 [cs, stat]*, Feb. 2018, Accessed: Jan. 26, 2022. [Online]. Available: [http://arxiv.org/abs/1802.02163](http://arxiv.org/abs/1802.02163)

[3]
F. Gilardi, C. R. Shipan, and B. Wüest, “Policy Diffusion: The Issue‐Definition Stage,” *American Journal of Political Science*, vol. 65, no. 1, pp. 21–35, Jan. 2021, doi: [10.1111/ajps.12521](https://doi.org/10.1111/ajps.12521).

[4]
P. Bojanowski, E. Grave, A. Joulin, and T. Mikolov, “Enriching Word Vectors with Subword Information,” *arXiv:1607.04606 [cs]*, Jun. 2017, Accessed: Dec. 20, 2021. [Online]. Available: [http://arxiv.org/abs/1607.04606](http://arxiv.org/abs/1607.04606)

[5]
M. E. Roberts, B. M. Stewart, and E. M. Airoldi, “A Model of Text for Experimentation in the Social Sciences,” *Journal of the American Statistical Association*, vol. 111, no. 515, pp. 988–1003, Jul. 2016, doi: [10.1080/01621459.2016.1141684](https://doi.org/10.1080/01621459.2016.1141684).

[6]
F. J. Boehmke *et al.*, “SPID: A New Database for Inferring Public Policy Innovativeness and Diffusion Networks,” *Policy Stud J*, vol. 48, no. 2, pp. 517–545, May 2020, doi: [10.1111/psj.12357](https://doi.org/10.1111/psj.12357).

[7]
M. E. Roberts, D. Tingley, B. M. Stewart, and E. M. Airoldi, “The Structural Topic Model and Applied Social Science,” p. 4.

[8]
J. Grimmer, M. E. Roberts, and B. M. Stewart, “Machine Learning for Social Science: An Agnostic Approach,” *Annual Review of Political Science*, vol. 24, no. 1, pp. 395–419, 2021, doi: [10.1146/annurev-polisci-053119-015921](https://doi.org/10.1146/annurev-polisci-053119-015921).

[9]
P. H. Luz De Araujo, T. De Campos, and filo, “Topic Modelling Brazilian Supreme Court Lawsuits,” Legal Knowledge and Information Systems, pp. 113–122, 2020, doi: 10.3233/FAIA200855.

