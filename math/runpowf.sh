./math/pow_test --dump-inputs ./powtest.in
../torch/torchbinary.py --op pow --file ./powtest.in
./math/pow_test --torchinductor torchinductorpow.bin --torcheager torcheagerpow.bin --verbose --quiet --color | less -R
