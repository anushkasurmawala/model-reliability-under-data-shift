# Dataset provenance

Adult, UCI Machine Learning Repository, Barry Becker and Ronny Kohavi.
https://archive.ics.uci.edu/dataset/2/adult
DOI: https://doi.org/10.24432/C5XW20
License: Creative Commons Attribution 4.0 (per UCI dataset page).

Downloaded from https://archive.ics.uci.edu/static/public/2/adult.zip.
Original files are retained in raw/ without changes. SHA-256 fingerprints are
recorded in ../reports/run_metadata.json. The runner can download the three
required original files if they are absent. No private data or credentials
are required.

UCI describes extraction from a 1994 Census database, with age > 16, income >
100, final weight > 1 and hours worked > 0. This selected historical sample
must not be treated as a current representative population.

We combine adult.data and adult.test because this project defines its own
population experiment, rather than comparing with published benchmark scores.
The original official partition is not interpreted as temporal.
