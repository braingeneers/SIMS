export TYPE=human_cortical && export NAME=human-cortical && envsubst < yaml/model.yaml | kubectl create -f -

export TYPE=mouse_cortical && export NAME=mouse-cortical && envsubst < yaml/model.yaml | kubectl create -f -

export TYPE=retina && export NAME=retina && envsubst < yaml/model.yaml | kubectl create -f -

export TYPE=dental && export NAME=dental && envsubst < yaml/model.yaml | kubectl create -f -