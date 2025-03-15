1. Trenujemy 200 modeli
   - 200 razy losowo dzielimy private dataset na dwie części (równe)
   - uczymy model na częsci 1. zbioru, a część 2. odrzucamy
   - mamy w ten sposób dla każdej próbi około 100 modeli które były uczone na tej próbce i 100 modeli, które nie były uczone na tej próbce
   - dla każdej próbki ustalamy jeden threshold dla wartości funkcji straty
  
Rozmkiny:
- jak trenujemy 200 modeli, to czy skorzystać z domyślnego resnet-18, czy może wziąć ich wytrenowany
- w pierwszej kolejności też zróbmy wersję "baseline", czyli jeden threshhold dla każdego obrazka
    - puszczamy ich model na ich "public" dataset, żeby jakoś wyznaczyć próg