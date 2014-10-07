#!/bin/sh
Pweave -f pandoc apsg.mdw
pandoc -o README.md -t markdown_github apsg.md
rm apsg.md

