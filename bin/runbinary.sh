complexops/binary_test --dump-inputs ./binarytest.in
for i in atan2 cldiv copysign dim div fldiv fmax fmin fmod hypot nextafter pow remainder root ; do
#for i in div ; do
  ../bin/torchbinary.py --op $i --file ./binarytest.in
done
#exit
for i in atan2 cldiv copysign dim div fldiv fmax fmin fmod hypot nextafter pow remainder root ; do
#for i in div ; do
  complexops/binary_test --$i --torchinductor torchinductor$i.bin --torcheager torcheager$i.bin --quiet --csv > $i.csv
  complexops/binary_test --$i --torchinductor torchinductor$i.bin --torcheager torcheager$i.bin --quiet --color | tee $i.out
  #complexops/binary_test --$i --torchinductor torchinductor$i.bin --torcheager torcheager$i.bin --quiet --color | less -R
  #complexops/binary_test --$i --torchinductor torchinductor$i.bin --torcheager torcheager$i.bin --color | less -R
done
