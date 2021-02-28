#!/bin/sh

mkdir /mit_data                         && \
cd /mit_data                          && \
curl -sS https://storage.googleapis.com/mitdb-1.0.0.physionet.org/mit-bih-arrhythmia-database-1.0.0.zip > file.zip && \
unzip file.zip                                  && \
rm file.zip