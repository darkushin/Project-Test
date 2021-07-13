#!/bin/bash
for i in {1..20}
do
  if ((i < 10))
  then
    curl -O "https://raw.githubusercontent.com/anchen1011/toflow/master/data/example/blur/0$i.png"
  else
    curl -O "https://raw.githubusercontent.com/anchen1011/toflow/master/data/example/blur/$i.png"
  fi
done