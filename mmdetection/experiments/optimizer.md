SGD -> Momentum -> RMSprop -> Adam

adam
lr=1e-3 weight_decay=1e-4 卒 0.6
lr=1e-2 weight_decay=1e-4 卒 0.0
lr=2e-4 weight_decay=1e-4 活 20.8

Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)

Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
betas,eps不用动
amsgrad=true iclr2018


AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

