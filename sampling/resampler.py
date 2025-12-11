# sampling/resampler.py

import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gammainccinv, logsumexp

def _kth_radius_tree(tree, X_query, k):
    dists, _ = tree.query(X_query, k=k, workers=-1)
    r_k = np.atleast_1d(dists) if k == 1 else dists[..., -1]
    return np.asarray(r_k, dtype=float)

def _calculate_log_density(X_scaled, k, d):
    N = X_scaled.shape[0]
    tree = cKDTree(X_scaled)
    k_eff = min(k + 1, max(2, N))
    r_k = _kth_radius_tree(tree, X_scaled, k_eff)
    r = np.maximum(r_k, 1e-12)
    return tree, -d * np.log(r)

def _target_concentration_log(X_scaled, logL_true, k, d, T_T):
    N = X_scaled.shape[0]
    tree, log_rho = _calculate_log_density(X_scaled, k, d)
    
    if np.isfinite(T_T):
        inv_T = 1.0 / T_T
        log_D = logsumexp(inv_T * logL_true - log_rho)
    else:
        log_D = logsumexp(-log_rho)
    
    log_c = np.log(N) - log_D
    return float(log_c), float(log_D)

def _flat_adjust_target_log_if_needed(log_target, N_S, N_new, d):
    a = 0.5 * d
    x_S = gammainccinv(a, (N_S - 1.0) / N_S)
    x_new = gammainccinv(a, (N_new - 1.0) / N_new)
    n_S = np.sqrt(2.0 * x_S)
    n_new = np.sqrt(2.0 * x_new)
    
    if n_new < n_S:
        return log_target + d * np.log(n_S / n_new)
    return log_target

def _compute_target_comparison(x_all, y_all, X_cand, L_cand_emul, x_scaler, k, d, T_T):
    Xs = x_scaler.transform(x_all)
    log_target_S, _ = _target_concentration_log(Xs, y_all, k, d, T_T)
    
    X_cand_scaled = x_scaler.transform(X_cand)
    
    if not np.isfinite(T_T):
        dummy_y_cand = np.zeros(X_cand.shape[0])
        log_target_new, _ = _target_concentration_log(X_cand_scaled, dummy_y_cand, k, d, T_T)
    else:
        log_target_new, _ = _target_concentration_log(X_cand_scaled, L_cand_emul, k, d, T_T)
    
    return max(log_target_S, log_target_new)

def _calculate_reweighting_probabilities(emul_logL, T_T, T_MC):
    inv_T_T = 0.0 if not np.isfinite(T_T) else (1.0 / T_T)
    inv_T_MC = 1.0 / T_MC
    logw = (inv_T_T - inv_T_MC) * emul_logL
    logw -= logsumexp(logw)
    return np.exp(logw)

def _weighted_candidates(samples, emul_logL, T_T, T_MC, N_take, rng):
    p = _calculate_reweighting_probabilities(emul_logL, T_T, T_MC)
    n_samples = min(N_take, samples.shape[0])
    return rng.choice(samples.shape[0], size=n_samples, replace=False, p=p)

def _create_likelihood_wrapper(likelihood, param_names):
    return lambda x: likelihood.loglkl({name: float(x[j]) for j, name in enumerate(param_names)})

def _evaluate_likelihood_serial(likelihood, X_batch, param_names):
    if hasattr(likelihood, 'loglkl_array'):
        return likelihood.loglkl_array(X_batch)
    
    return np.array([
        likelihood.loglkl({name: float(X_batch[i, j]) for j, name in enumerate(param_names)})
        for i in range(len(X_batch))
    ])

def _evaluate_likelihood_batch(likelihood, X_batch, param_names, use_mpi=False):
    if use_mpi:
        from utils.mpi_utils import parallel_evaluate_likelihood
        likelihood_wrapper = _create_likelihood_wrapper(likelihood, param_names)
        return parallel_evaluate_likelihood(X_batch, likelihood_wrapper, description="resampling batch")
    
    return _evaluate_likelihood_serial(likelihood, X_batch, param_names)

def _handle_random_resampling(cfg, samples, true_likelihood, param_names, use_mpi, return_metrics, start_time, d):
    from utils.mpi_utils import is_master, get_communicator
    
    if is_master():
        rng = np.random.default_rng()
        idx = rng.choice(samples.shape[0], size=cfg.n_candidates, replace=False)
        x_new = samples[idx]
    else:
        x_new = None
    
    if use_mpi:
        comm = get_communicator()
        x_new = comm.bcast(x_new, root=0)
    
    y_new = _evaluate_likelihood_batch(true_likelihood, x_new, param_names, use_mpi=use_mpi)
    
    if is_master():
        print(f"   Selected {len(x_new)} random samples")
        print(f"   Log-likelihood range: [{y_new.min():.2f}, {y_new.max():.2f}]")
        print(f"Random resampling complete: returning x_new.shape={x_new.shape}, y_new.shape={y_new.shape}")
        
        if return_metrics:
            resampling_metrics = {
                'candidates_processed': cfg.n_candidates,
                'rejected_emulated': 0,
                'rejected_true': 0,
                'accepted': len(x_new),
                'resampling_time': time.time() - start_time,
                'n_initial_samples': cfg.n_samples
            }
            return x_new, y_new, resampling_metrics
        return x_new, y_new
    
    if return_metrics:
        resampling_metrics = {
            'candidates_processed': 0,
            'rejected_emulated': 0,
            'rejected_true': 0,
            'accepted': 0,
            'resampling_time': 0,
            'n_initial_samples': 0
        }
        return np.empty((0, d)), np.empty(0), resampling_metrics
    return np.empty((0, d)), np.empty(0)

def generate_resamples(cfg, samples, loglkls, true_likelihood, x_all, y_all, x_scaler, return_metrics=False, use_mpi=False):
    from utils.mpi_utils import is_master, get_communicator, barrier
    
    start_time = time.time() if return_metrics else None
    print(f"\nStarting resampling with strategy: {cfg.rs_strategy}")
    print(f"   Requested {cfg.n_candidates} new samples from {samples.shape[0]} MCMC samples")
    
    param_names = true_likelihood.varying_param_names
    d = x_all.shape[1]
    
    if cfg.rs_strategy == 'random':
        print(f"Using random resampling...")
        return _handle_random_resampling(cfg, samples, true_likelihood, param_names, use_mpi, return_metrics, start_time, d)
    
    if cfg.rs_strategy == 'adaptive':
        print(f"Starting adaptive resampling...")
        k = cfg.k_NN
        T_T = cfg.temperature_training
        T_MC = cfg.temperature
        rng = np.random.default_rng()

        if is_master():
            print(f"   Training data: x_all.shape = {x_all.shape}, y_all.shape = {y_all.shape}")
            print(f"   Training log-lkl range: [{y_all.min():.2f}, {y_all.max():.2f}], mean: {y_all.mean():.2f}")
            print(f"   Parameters: d={d}, k={k}, T_T={T_T}, T_MC={T_MC}")

            Xs = x_scaler.transform(x_all)
            log_target, log_D = _target_concentration_log(Xs, y_all, k, d, T_T)
            print(f"   Computed log_target: {log_target:.6f}")

            logL_chain_emul = loglkls
            cand_idx = _weighted_candidates(samples, logL_chain_emul, T_T, T_MC, cfg.n_candidates, rng)
            X_cand = samples[cand_idx]
            L_cand_emul = logL_chain_emul[cand_idx]

            print(f"   MCMC samples: {samples.shape}, loglkl range: [{loglkls.min():.2f}, {loglkls.max():.2f}]")
            print(f"   Selected {len(cand_idx)} candidates, emul_loglkl range: [{L_cand_emul.min():.2f}, {L_cand_emul.max():.2f}]")

            log_target_S, _ = _target_concentration_log(x_scaler.transform(x_all), y_all, k, d, T_T)
            log_target_new = _compute_target_comparison(x_all, y_all, X_cand, L_cand_emul, x_scaler, k, d, T_T)
            log_target = log_target_new
            print(f"   Step 3 comparison: log_c_S = {log_target_S:.6f}, log_c_comparison = {log_target_new:.6f}")
            print(f"   Using larger target: {log_target:.6f}")

            tree = cKDTree(Xs)
            accepted_X = []
            accepted_y = []
            since_rebuild = 0
            
            all_accepted_X = []
            all_accepted_y = []

            inv_T_T = 0.0 if not np.isfinite(T_T) else (1.0 / T_T)
            N_cur = x_all.shape[0]
            
            n_rejected_emul = 0
            n_rejected_true = 0
            n_accepted = 0
            
            print(f"   Processing {len(X_cand)} candidates (inv_T_T = {inv_T_T})...")
            detailed_print_count = min(3, len(X_cand))

            y_all_cur = np.array(y_all, dtype=float)
        else:
            all_accepted_X = []
            all_accepted_y = []
            n_rejected_emul = 0
            n_rejected_true = 0
            n_accepted = 0
            X_cand = None

        batch_size_config = min(1000, cfg.n_candidates) if is_master() else 1000
        batch_start = 0
        
        while True:
            if is_master():
                n_candidates_total = len(X_cand)
                if batch_start >= n_candidates_total:
                    batch_size = 0
                    X_batch = None
                    L_batch_emul = None
                else:
                    batch_end = min(batch_start + batch_size_config, n_candidates_total)
                    batch_size = batch_end - batch_start
                    X_batch = X_cand[batch_start:batch_end]
                    L_batch_emul = L_cand_emul[batch_start:batch_end]
            else:
                batch_size = None
                X_batch = None
                L_batch_emul = None
            
            if use_mpi:
                comm = get_communicator()
                batch_size = comm.bcast(batch_size, root=0)
            
            if batch_size == 0:
                break
            
            if is_master():
                X_batch_scaled = x_scaler.transform(X_batch)
                r_k_batch = _kth_radius_tree(tree, X_batch_scaled, k)
                r_batch = np.maximum(r_k_batch, 1e-12)
                log_rho_batch = -d * np.log(r_batch)
                
                if inv_T_T == 0.0:
                    log_val_emul_batch = log_rho_batch
                else:
                    log_val_emul_batch = log_rho_batch - L_batch_emul * inv_T_T
                
                emul_pass_mask = log_val_emul_batch < log_target
                n_rejected_emul += np.sum(~emul_pass_mask)
                
                if np.any(emul_pass_mask):
                    X_emul_pass = X_batch[emul_pass_mask]
                    log_rho_emul_pass = log_rho_batch[emul_pass_mask]
                else:
                    X_emul_pass = np.empty((0, d))
                    log_rho_emul_pass = np.empty(0)
            else:
                X_emul_pass = None
                log_rho_emul_pass = None
            
            if use_mpi:
                X_emul_pass = comm.bcast(X_emul_pass, root=0)
            
            if len(X_emul_pass) > 0:
                y_true_batch = _evaluate_likelihood_batch(
                    true_likelihood, X_emul_pass, param_names, use_mpi=use_mpi
                )
            else:
                y_true_batch = np.empty(0) if is_master() else np.empty(0)
            
            if is_master() and len(X_emul_pass) > 0:
                if inv_T_T == 0.0:
                    log_val_true_batch = log_rho_emul_pass
                else:
                    log_val_true_batch = log_rho_emul_pass - y_true_batch * inv_T_T
                
                true_pass_mask = log_val_true_batch < log_target
                n_rejected_true += np.sum(~true_pass_mask)
                
                if np.any(true_pass_mask):
                    X_accepted_batch = X_emul_pass[true_pass_mask]
                    y_accepted_batch = y_true_batch[true_pass_mask]
                    
                    accepted_X.extend(X_accepted_batch)
                    accepted_y.extend(y_accepted_batch)
                    all_accepted_X.extend(X_accepted_batch)
                    all_accepted_y.extend(y_accepted_batch)
                    
                    n_accepted += len(X_accepted_batch)
                    
                    if batch_start < detailed_print_count:
                        for j in range(min(detailed_print_count - batch_start, len(X_batch))):
                            idx = batch_start + j
                            r_k = r_k_batch[j]
                            log_rho_p = log_rho_batch[j]
                            logL_emul_p = L_batch_emul[j]
                            log_val_emul = log_rho_p if inv_T_T == 0.0 else (log_rho_p - logL_emul_p * inv_T_T)
                            print(f"     Candidate {idx}: r_k={r_k:.6f}, log_rho_p={log_rho_p:.2f}, logL_emul_p={logL_emul_p:.2f}")
                            print(f"                    log_val_emul={log_val_emul:.2f}, log_target={log_target:.2f}")
                            
                            if emul_pass_mask[j]:
                                print(f"                    -> Passed emulated, evaluating true likelihood...")
                            else:
                                print(f"                    -> REJECTED (emulated criterion)")
                    
                    if np.isfinite(T_T):
                        for idx_in_batch, y_true in enumerate(y_accepted_batch):
                            log_rho_p = log_rho_emul_pass[true_pass_mask][idx_in_batch]
                            log_term_true = (-log_rho_p) if inv_T_T == 0.0 else (inv_T_T * y_true - log_rho_p)
                            log_D = logsumexp([log_D, log_term_true])
                            N_cur += 1
                            log_target = np.log(N_cur) - log_D
            
            if is_master():
                since_rebuild += batch_size
                
                if since_rebuild >= int(max(1, cfg.update_freq)) and len(accepted_X) > 0:
                    x_all = np.vstack([x_all, np.array(accepted_X, dtype=float)])
                    y_all_cur = np.concatenate([y_all_cur, np.array(accepted_y, dtype=float)])
                    Xs = x_scaler.transform(x_all)
                    tree = cKDTree(Xs)

                    log_target_rebuild, log_D = _target_concentration_log(Xs, y_all_cur, k, d, T_T)
                    N_cur = x_all.shape[0]
                    
                    remaining_start = batch_start + batch_size
                    if remaining_start < len(X_cand):
                        X_remaining = X_cand[remaining_start:]
                        L_remaining_emul = L_cand_emul[remaining_start:]
                        log_target = _compute_target_comparison(x_all, y_all_cur, X_remaining, L_remaining_emul, x_scaler, k, d, T_T)
                    else:
                        log_target = log_target_rebuild

                    accepted_X.clear()
                    accepted_y.clear()
                    since_rebuild = 0
                
                batch_end_for_reporting = batch_start + batch_size
                if batch_end_for_reporting % 500 == 0 or batch_end_for_reporting == len(X_cand):
                    current_acceptance = n_accepted / batch_end_for_reporting if batch_end_for_reporting > 0 else 0
                    print(f"   Progress: {batch_end_for_reporting}/{len(X_cand)} processed, {n_accepted} accepted ({current_acceptance:.3f} rate)")
                
                batch_start += batch_size

        if is_master():
            print(f"   Final Results:")
            print(f"      Candidates processed: {len(X_cand)}")
            print(f"      Rejected at emulated stage: {n_rejected_emul}")
            print(f"      Rejected at true stage: {n_rejected_true}")
            print(f"      Accepted: {n_accepted}")
            print(f"      Final acceptance rate: {n_accepted / len(X_cand):.4f}")
            
            if n_accepted == 0:
                print(f"   WARNING: No points accepted! Potential issues:")
                if n_rejected_emul == len(X_cand):
                    print(f"      - All points failed emulated criterion (log_target too restrictive)")
                    print(f"      - Consider increasing temperature_training or decreasing k_NN")
                elif n_rejected_emul + n_rejected_true == len(X_cand):
                    print(f"      - Points passed emulated but failed true criterion")
                    print(f"      - Possible emulator inaccuracy or likelihood issues")
            else:
                print(f"   Successfully accepted {n_accepted} new training points")

            if accepted_X:
                x_all = np.vstack([x_all, np.array(accepted_X, dtype=float)])
                y_all_cur = np.concatenate([y_all_cur, np.array(accepted_y, dtype=float)])

            x_new = np.array(all_accepted_X, dtype=float) if all_accepted_X else np.empty((0, d), dtype=float)
            y_new = np.array(all_accepted_y, dtype=float) if all_accepted_y else np.empty((0,), dtype=float)
            
            print(f"Resampling complete: returning x_new.shape={x_new.shape}, y_new.shape={y_new.shape}")
            if x_new.shape[0] > 0:
                print(f"   New training log-lkl range: [{y_new.min():.2f}, {y_new.max():.2f}]")
            
            if return_metrics:
                resampling_time = time.time() - start_time
                resampling_metrics = {
                    'candidates_processed': len(X_cand),
                    'rejected_emulated': n_rejected_emul,
                    'rejected_true': n_rejected_true,
                    'accepted': n_accepted,
                    'resampling_time': resampling_time,
                    'n_initial_samples': cfg.n_samples
                }
                return x_new, y_new, resampling_metrics
            else:
                return x_new, y_new
        else:
            if return_metrics:
                resampling_metrics = {
                    'candidates_processed': 0,
                    'rejected_emulated': 0,
                    'rejected_true': 0,
                    'accepted': 0,
                    'resampling_time': 0,
                    'n_initial_samples': 0
                }
                return np.empty((0, d)), np.empty(0), resampling_metrics
            else:
                return np.empty((0, d)), np.empty(0)

    raise ValueError(f"Unknown resampling strategy: {cfg.rs_strategy}")
