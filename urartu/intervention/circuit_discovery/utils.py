def schedule_epoch_lambda(epoch, lambda_0, max_times=1., min_times=1., 
                          n_epoch_warmup=0, n_epoch_cooldown=0):

    if epoch < n_epoch_warmup:
        return lambda_0  + lambda_0 * (max_times - 1) * epoch / n_epoch_warmup
        
    elif epoch < n_epoch_warmup + n_epoch_cooldown:
        return lambda_0 * max_times - lambda_0 * (max_times - min_times) * (epoch - n_epoch_warmup) / n_epoch_cooldown
        
    else:
        return lambda_0 * min_times
