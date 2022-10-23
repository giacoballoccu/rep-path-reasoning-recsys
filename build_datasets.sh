for dataset in ml1m lfm1m ;
  do
  	for model in cke cfkg kgat fm nfm bprmf pgpr ucpr cafe ;
		do
			python3 map_dataset.py --data $dataset --model $model
		done    	
  done
