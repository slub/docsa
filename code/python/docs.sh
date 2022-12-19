#!/bin/bash

rm -rf ../../docs/python/slub_docsa

pdoc -o ../../docs/python src/slub_docsa --docformat numpy --footer-text "slub_docsa v0.1.0.dev1"