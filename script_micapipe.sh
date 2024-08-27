bids=/home/pabaua/dev_mni/data/sansan_bids

out=/home/pabaua/dev_mni/results

fs_lic=/home/pabaua/dev_tpil/data/Freesurfer/license.txt

tmp=/home/pabaua/dev_mni/tmp

sub=sansan

ses=01

echo ${bids}

docker run -ti --rm \
    -v ${bids}:/bids \
    -v ${out}:/out \
    -v ${tmp}:/tmp \
    -v ${fs_lic}:/opt/licence.txt \
    micalab/micapipe:v0.2.3 \
    -bids /bids -out /out -fs_licence /opt/licence.txt -threads 10 \
    -sub $sub \
    -proc_structural
