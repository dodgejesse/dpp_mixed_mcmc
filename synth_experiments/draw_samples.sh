for i in `seq 0 600`; do
    python discrepancy.py ${i}_1
    echo "finished ${i}"
done


python compute_discrep_from_saved_data.py
