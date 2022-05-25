C=1
# for P in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 
# do
#     export PROP=${P} && export COUNT=0${C} && envsubst < yaml/ablation.yaml | kubectl create -f -
#     (( C++ ))
# done 

C=1
for P in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
do
    export PROP=${P} && export COUNT=${C} && envsubst < yaml/ablation.yaml | kubectl create -f -
    (( C++ ))
done 
