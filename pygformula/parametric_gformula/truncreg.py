import numpy as np
from scipy.stats import norm
from scipy.optimize import Bounds, minimize
import re

def truncreg(formula, data, point, direction, scaled=False, iterlim=50):
    if direction not in ["left", "right"]:
        raise ValueError("'direction' must be 'left' or 'right'")
    
    x_vars = re.split(r'[*+]', re.search(r'~\s*(.+)', formula).group(1).strip())
    x_vars = [var.strip() for var in x_vars]
    x = data[[var for var in x_vars if var]]
    y = data[formula.split('~')[0].strip()]
    
    intercept = np.ones((x.shape[0], 1))
    x = np.hstack((intercept, x))
    
    if direction == "left" and any(y < point):
        raise ValueError("Data non-truncated, contains observations < '{}'".format(point))
    if direction == "right" and any(y > point):
        raise ValueError("Data non-truncated, contains observations > '{}'".format(point))

    def maxLikTruncreg(param, x, y, point, direction, scaled=scaled):
        epsilon = 1e-8
        sign = 1 if direction == "left" else -1
        beta = param[:-1]
        sigma = param[-1]
        bX = np.dot(x, beta)
        
        if not scaled:
            resid = y - bX
            trunc = bX - point
    
            #Potential Overflow Handling
            exp_argument = norm.logpdf(sign * trunc / (sigma + epsilon)) - norm.logcdf(sign * trunc / (sigma + epsilon))
            exp_argument = np.clip(exp_argument, a_min=None, a_max=700)
            mills = np.exp(exp_argument)
    
            lnL = -np.log(sigma + epsilon) + norm.logpdf(resid / (sigma + epsilon)) - norm.logcdf(sign * trunc / (sigma + epsilon))
            logLik = lnL.sum() if isinstance(lnL, np.ndarray) and lnL.ndim > 0 else lnL
    
            # Gradient calculation
            gbX = resid / (sigma ** 2 + epsilon) - sign * mills / (sigma + epsilon)
            gsigma = -1 / (sigma + epsilon) + resid ** 2 / (sigma + epsilon) ** 3 + sign * mills * trunc / (sigma + epsilon) ** 2
            gradient = np.concatenate([gbX, gsigma])
    
            # Hessian calculation
            bb = -1 / (sigma ** 2 + epsilon) + mills * (sign * trunc / (sigma + epsilon) + mills) / (sigma + epsilon) ** 2
            ss = 1 / (sigma + epsilon) ** 2 - 3 * resid ** 2 / (sigma + epsilon) ** 4 - 2 * mills * sign * trunc / (sigma + epsilon) ** 3 + \
                 mills * (sign * trunc / (sigma + epsilon) + mills) * trunc ** 2 / (sigma + epsilon) ** 4
            bs = -2 * resid / (sigma + epsilon) ** 3 + sign * mills / (sigma + epsilon) ** 2 - \
                 mills * (mills + sign * trunc / (sigma + epsilon)) * trunc / (sigma + epsilon) ** 3
            bs = np.array(bs)
    
            bb_matrix = np.dot(x.T, np.reshape(bb, (-1, 1)) * x)
            bs_vector = np.sum(np.reshape(bs, (-1, 1)) * x, axis=0)
            bs_vector = np.reshape(bs_vector, (-1, 1))
            ss_scalar = np.array([[np.sum(ss)]])
    
            hessian = np.block([[bb_matrix, bs_vector],
                            [bs_vector.T, ss_scalar]])

        else:
            exp_argument = norm.logpdf(sign * (bX - point / (sigma + epsilon))) - norm.logcdf(sign * (bX - point / (sigma + epsilon)))
            exp_argument = np.clip(exp_argument, a_min=None, a_max=700)
            mills = np.exp(exp_argument)
            
            lnL = -np.log(sigma + epsilon) + norm.logpdf(y / (sigma + epsilon) - bX) - norm.logcdf(sign * (bX - point / (sigma + epsilon)))
            logLik = lnL.sum() if isinstance(lnL, np.ndarray) and lnL.ndim > 0 else lnL
            
            #Gradient calculation
            gbX = (y / sigma - bX + epsilon) - mills * sign
            gsigma = -1 / (sigma + epsilon) + (y / sigma + epsilon - bX) * y /(sigma + epsilon) ** 2 - sign * mills * point/(sigma + epsilon) ** 2
            gradient = np.concatenate([gbX, gsigma])
            
            #Hessian Calculation
            bb = -1 + mills * (mills + sign * (bX - point / (sigma + epsilon)))
            bs = -y / sigma ** 2 + (mills + sign * (bX - point / (sigma + epsilon))) * mills * point / (sigma + epsilon) ** 2
            ss = 1 / (sigma + epsilon) ** 2 - 3 * y ** 2 / (sigma + epsilon) ** 4 + 2 * bX * y / (sigma + epsilon) ** 3 + \
                mills * (mills + sign * (bX - point / (sigma + epsilon))) * point ** 2 / (sigma + epsilon) ** 4 + \
                2 * sign * mills * point / (sigma + epsilon) ** 3
            bs = np.array(bs)
            
            bb_matrix = np.dot(x.T, np.reshape(bb, (-1, 1)) * x)
            bs_vector = np.sum(np.reshape(bs, (-1, 1)) * x, axis=0)
            bs_vector = np.reshape(bs_vector, (-1, 1))
            ss_scalar = np.array([[np.sum(ss)]])
            
            hessian = np.block([[bb_matrix, bs_vector],
                            [bs_vector.T, ss_scalar]])

        return {
            'logLik': logLik, 
            'gradient': gradient, 
            'hessian': hessian,
        }

    def objective(param):
        result = maxLikTruncreg(param, x, y, point, direction)
        return -result['logLik']

    linmod = np.linalg.lstsq(x, y, rcond=None)
    start_beta = linmod[0]
    start_sigma = np.array([np.std(linmod[1])])
    start = np.concatenate([start_beta, start_sigma])

    lower_bounds = [np.NINF] * (len(start) - 1) + [1e-4]
    upper_bounds = [np.inf] * len(start)
    bounds = Bounds(lower_bounds, upper_bounds)

    result = minimize(objective, start, method='L-BFGS-B', bounds=bounds, options={'maxiter': iterlim})

    return result
