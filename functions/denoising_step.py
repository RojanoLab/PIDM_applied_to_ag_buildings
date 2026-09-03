import torch

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def ddim_steps(x, seq, model, b, **kwargs):
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    dx_func = kwargs.get('dx_func', None)
    clamp_func = kwargs.get('clamp_func', None)
    cache = kwargs.get('cache', False)

    logger = kwargs.get('logger', None)
    if logger is not None:
        logger.update(x=xs[-1])

    for i, j in zip(reversed(seq), reversed(seq_next)):
        with torch.no_grad():
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c2 = (1 - at_next).sqrt()
        if dx_func is not None:
            dx = dx_func(xt)
        else:
            dx = 0
        with torch.no_grad():
            xt_next = at_next.sqrt() * x0_t + c2 * et - dx
            if clamp_func is not None:
                xt_next = clamp_func(xt_next)
            xs.append(xt_next.to('cpu'))

        if logger is not None:
            logger.update(x=xs[-1])

        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]

    return xs, x0_preds


def ddpm_steps(x, seq, model, b,  **kwargs):
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    xs = [x]
    x0_preds = []
    betas = b
    dx_func = kwargs.get('dx_func', None)
    cache = kwargs.get('cache', False)
    clamp_func = kwargs.get('clamp_func', None)

    for i, j in zip(reversed(seq), reversed(seq_next)):
        with torch.no_grad():

            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()

        if dx_func is not None:
            dx = dx_func(x)
        else:
            dx = 0
        with torch.no_grad():
            sample = mean + mask * torch.exp(0.5 * logvar) * noise - dx
            if clamp_func is not None:
                sample = clamp_func(sample)
            xs.append(sample.to('cpu'))
        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]

    return xs, x0_preds


def guided_ddpm_steps(x, seq, model, b,  **kwargs):
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    xs = [x]
    x0_preds = []
    betas = b
    dx_func = kwargs.get('dx_func', None)
    if dx_func is None:
        raise ValueError('dx_func is required for guided denoising')
    clamp_func = kwargs.get('clamp_func', None)
    cache = kwargs.get('cache', False)
    w = kwargs.get('w', 3.0)

    for i, j in zip(reversed(seq), reversed(seq_next)):
        with torch.no_grad():

            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

        dx = dx_func(x)
        with torch.no_grad():

            output = (w+1)*model(x, t.float(), dx)-w*model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()


        with torch.no_grad():
            sample = mean + mask * torch.exp(0.5 * logvar) * noise - dx
            if clamp_func is not None:
                sample = clamp_func(sample)
            xs.append(sample.to('cpu'))
        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]

    return xs, x0_preds


def guided_ddim_steps(x, seq, model, b, **kwargs):
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    
    dx_func = kwargs.get('dx_func', None)
    if dx_func is None:
        raise ValueError('dx_func is required for guided denoising')

    input_fn = kwargs.get('input_fn', None)

    cond = kwargs.get('dx', None)
    if cond is None:
        raise ValueError('dx (condición de entrada) is required for guided denoising')

    clamp_func = kwargs.get('clamp_func', None)
    cache = kwargs.get('cache', False)
    w = kwargs.get('w', 3.0)
    w_min = kwargs.get('w_min', w)
    dx_scale = kwargs.get('dx_scale', 1.0)
    logger = kwargs.get('logger', None)


    if logger is not None:
        logger.update(x=xs[-1].to('cuda'))    

    total_steps = max(1, len(seq))
    for step_idx, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
        with torch.no_grad():
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

        # dx_func returns (dx, dx_scaled).
        # dx is the unscaled physical-unit condition expected by the model.
        # dx_scaled is used only for the DDIM update in z-score space.
        dx, dx_scaled = dx_func(xt)

        # Keep the full 6-channel conditioning tensor expected by the
        # conditional model ([u, v] concatenated). Do not crop it to 3 channels.
        if dx.shape[1] != 6:
            raise ValueError(f"Expected 6-channel conditioning tensor for the conditional model, got shape {tuple(dx.shape)}")

        # ── DIAGNÓSTICO ──
        if i in list(reversed(list(seq)))[:3]:
            dx_norm = dx_scaled.norm().item()
            dx_mean_abs = dx_scaled.abs().mean().item()
            dx_max = dx_scaled.abs().max().item()
            print(f"[DIAG | t={int(i):4d}] dx_scaled norm={dx_norm:.6f} | mean_abs={dx_mean_abs:.6f} | max={dx_max:.6f}")

        with torch.no_grad():
            model_input = input_fn(xt) if input_fn is not None else xt

            # Linearly decay guidance strength from w to w_min across reverse steps.
            progress = step_idx / max(1, total_steps - 1)
            w_t = w + (w_min - w) * progress

            # The conditional model was trained with the physical residual
            # guidance dx as an additional condition.
            et = (w_t + 1) * model(model_input, t, dx) - w_t * model(model_input, t)

            # ── DIAGNÓSTICO ──
            if i in list(reversed(list(seq)))[:3]:
                et_dm   = model(model_input, t)
                et_pidm = (w+1)*model(model_input, t, dx) - w*model(model_input, t)
                diff_et = (et_pidm - et_dm).abs().mean().item()
                print(f"[DIAG | t={int(i):4d}] |et_pidm - et_dm| mean = {diff_et:.8f}")

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c2 = (1 - at_next).sqrt()

        with torch.no_grad():
            # 4. Subtract the physical guidance dx_scaled at the DDIM step
            xt_next = at_next.sqrt() * x0_t + c2 * et - (dx_scaled * dx_scale)
            if clamp_func is not None:
                xt_next = clamp_func(xt_next)
            
            xs.append(xt_next.to('cpu'))


        if logger is not None:
            logger.update(x=xs[-1].to('cuda'))    


        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]

    return xs, x0_preds
