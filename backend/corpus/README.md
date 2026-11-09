# Corpus snapshot

`chunks.jsonl.gz` is the library the evaluation ran on: 23,507 passages fetched from the
[Sefaria API](https://developers.sefaria.org/) on 9 November 2026 by `python -m chavruta.build_corpus`.
Each line is one natural unit (a verse, a mishnah, a Talmud segment, one comment of Rashi or
Tosafot, a halakhah, a seif) with its reference, language, version title and license, as the API
reported them.

Only versions whose license is Public Domain, CC0, CC-BY or CC-BY-SA were kept. Docker ingests this
file so a first run does not refetch 1,000 sections; the dashboard still shows the license of
every version Sefaria has for each work, from a live survey of the API.

| Work | Hebrew version | License | English version | License |
|---|---|---|---|---|
| Genesis–Deuteronomy | Tanach with Ta'amei Hamikra | Public Domain | The Holy Scriptures: A New Translation (JPS 1917) | Public Domain |
| Mishnah Berakhot | Torat Emet 357 | Public Domain | Mishnah Yomit by Dr. Joshua Kulp | CC-BY |
| Berakhot (Bavli) | Wikisource Talmud Bavli | CC-BY-SA | Tractate Berakot by A. Cohen, Cambridge University Press, 1921 | Public Domain |
| Rashi on Berakhot | Vilna Edition | Public Domain | – | |
| Tosafot on Berakhot | Vilna Edition | Public Domain | Tosafot, Translated by Jan Buckler | CC-BY |
| Mishneh Torah: Reading the Shema; Prayer and the Priestly Blessing; Blessings | Wikisource Mishneh Torah | CC-BY-SA | The Mishneh Torah by Maimonides, trans. Moses Hyamson, 1937–1949 | Public Domain |
| Shulchan Arukh, Orach Chayim 1–241 | Torat Emet 363 | Public Domain | – | |

Attribution: texts via Sefaria (sefaria.org). CC-BY texts: Mishnah Yomit by Dr. Joshua Kulp;
Tosafot translated by Jan Buckler. CC-BY-SA texts (Wikisource Talmud Bavli, Wikisource Mishneh
Torah) are redistributed here under the same license, CC BY-SA. This snapshot is offered under the
licenses of the texts it contains; the project's code license does not apply to it.
