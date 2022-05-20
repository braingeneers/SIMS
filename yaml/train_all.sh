export TYPE=human_cortical && export NAME=human-cortical-big-batch && envsubst < yaml/model.yaml | kubectl create -f -
export TYPE=mouse_cortical && export NAME=mouse-cortical-big-batch && envsubst < yaml/model.yaml | kubectl create -f -
export TYPE=retina && export NAME=retina-big-batch && envsubst < yaml/model.yaml | kubectl create -f -
export TYPE=dental && export NAME=dental-big-batch && envsubst < yaml/model.yaml | kubectl create -f -
export TYPE=pancreas && export NAME=pancreas-big-batch && envsubst < yaml/model.yaml | kubectl create -f -
export TYPE=mostajo_mouse && export NAME=mostajo-mouse-big-batch && envsubst < yaml/model.yaml | kubectl create -f -