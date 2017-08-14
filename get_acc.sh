#!/bin/bash

result_file=./result.mlf


HLEd -A -T 1 -X rec -i temp.mlf ./timit2_39.led $result_file
mv temp.mlf $result_file

oldsize=""
while newsize=`wc -c $result_file`; [ "$oldsize" != "$newsize" ]; do
   oldsize=$newsize
   HLEd -X rec -i temp.mlf ./mergesil.led $result_file
   mv temp.mlf $result_file
done

#test accuarcy
HResults -I ./ref39.mlf /dev/null $result_file
