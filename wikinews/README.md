# Multilingual WikiNews dataset

News contents were scraped out of all 20,307 English WikiNews articles published as of Sep 20, 2022, and non-English news articles that are listed as foreign pages of the English WikiNews article.
Out of all 20,307 English WikiNews pages scraped, 5240 pages have links to WikiNews in other languages. In total, 9960 foreign News titles paired to those English pages were retrieved (ave. 2 foreign links per page).

This dataset includes 15,200 multilingual WikiNews contents in 33 languages (with 9,960 non-English news + 5240 English news). 
List of languages are: Spanish, French, German, Portuguese, Polish, Italian, Chinese, Russian, Japanese, Dutch, Swedish, Tamil, Serbian, Czech, Catalan, Hebrew, Turkish, Finnish, Esperanto, Greek, Hungarian, Ukrainian, Norwegian, Arabic, Persian, Korean, Romanian, Bulgarian, Bosnian, Limburgish, Albanian, Thai.

## Weakly aligned multilingual pararell sentence datasets
Weakly aligned multilingual pararell sentence datasets can be constructed by comparing the titles and contents of the WikiNews pages that are linked to the same English WikiNews page (in the dataset, they have the same pageid).
Following is the example case where titles of the same pageid are retrieved.

| News title                                                    | Language | type              |
|---------------------------------------------------------------|----------|-------------------|
| Bomb blast in Delhi kills 12, injures 62                      | English  | title      |
| چندین کشته بر اثر انفجار بمب در مقابل دادگاه عالی هند        | Farsi    | title|
| 9 נהרגו בפיגוע מחוץ לבית המשפט העליון של הודו               | Hebrew   | title|
| У Индији 11 мртвих, 64 повређених                             | Serbian  | title|
| தில்லி உயர்நீதிமன்றத்தில் குண்டு வெடிப்பு, 10 பேர் உயிரிழப்பு | Tamil    | title|


## Example raw datasets

|   | title                                                       | pageid | categories                                         | lang | url                                                                                      | text                                                       | date                        | type            |
|---|-------------------------------------------------------------|--------|----------------------------------------------------|------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------|-----------------------------|-----------------|
| 0 | 'Bloody Sunday Inquiry' publishes report into ...           | 191513 | [Northern Ireland, Martin McGuinness, Politics...] | en   | https://en.wikinews.org/wiki/%27Bloody_Sunday_...                                       | [On Tuesday, the "Bloody Sunday Inquiry" publi...          | 2010-06-17                 | title           |
| 1 | 1972 ”இரத்த ஞாயிறு” படுகொலைகள் தொடர்பில் பிரித...           | 191513 | [Northern Ireland, Martin McGuinness, Politics...] | ta   | https://ta.wikinews.org/wiki/1972_%E2%80%9D%E0...                                        | [வடக்கு அயர்லாந்தில் 38 ஆண்டுகளுக்கு முன்னர் இ...   | வியாழன், சூன் 17, 2010 | interlang link |
| 2 | 'Very serious': Chinese government releases co...           | 232226 | [China, December 30, 2010, Politics and confli...] | en   | https://en.wikinews.org/wiki/%27Very_serious%2...                                       | [A report by the Chinese government states cor...         | 2010-12-30                 | title           |
| 3 | Čína připustila, že tamní korupce je vážný pro...           | 232226 | [China, December 30, 2010, Politics and confli...] | cs   | https://cs.wikinews.org/wiki/%C4%8C%C3%ADna_p%...                                        | [Zpráva čínské vlády připouští, že korupce v z... | Středa 29. prosince 2010 | interlang link |
| 4 | China admite que la corrupción en el país es '...           | 232226 | [China, December 30, 2010, Politics and confli...] | es   | https://es.wikinews.org/wiki/China_admite_que_...                                       | [29 de diciembre de 2010Beijing, China —, Un r... | None                        | interlang link |


## Variables

Each data point includes following variables:

| Field Name                        | Description                                                                                                                                           |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| title                             | WikiNews article title                                                                                                                               |
| pageid                             | pageid defined by the English WikiNews article. Data with the same pageid corresponds to the same news event linked together.             |
| categories                         | list of topics defined by WikiNews. All pages have at least one topic from [Crime and law, Culture and entertainment, Disasters and accidents, Economy and business, Education, Environment, Heath, Obituaries, Politics and conflicts, Science and technology, Sports, Wackynews, Weather] |
| text                              | content of the article. Some foreign pages have news titles but no content. For those, text is left empty.                                        |
| lang                      | languages of the article (WP code, check [here](https://en.wikipedia.org/wiki/List_of_Wikipedias#Lists) for lists )                                                                                        |
| url                               | articles' URL                                                                                                                                      |
| date                              | date of publish in YYYY-MM-DD for English pages. Dates in foreign pages were left as it is. To get a date with YYYY-MM-DD format, look for a English page with the same pageid.           |
