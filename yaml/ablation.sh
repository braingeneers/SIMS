COUNT=1
for P in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
do
    export PROP=${P} && export COUNT=${COUNT} && envsubst < yaml/ablation.yaml | kubectl create -f -
    (( COUNT++ ))
done 
