#include <iostream>
#include <chrono>
#include <Eigen/Eigen>
#include <unistd.h>
#include <fstream>

#define timestamp() std::chrono::high_resolution_clock::now()
#define time_elapsed_ms(t1, t0) std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()
#define time_elapsed_us(t1, t0) std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()
#define time_elapsed_s(t1, t0) std::chrono::duration<double>(t1-t0).count()

#define NEWTON_MAX_ITERS 5
#define NEWTON_TOLERANCE 1e-10
#define NUMERICAL_JACOBIAN_EPS 1e-6


/*////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                              numerical jacobian BDF
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////*/

Eigen::MatrixXd numerical_jacobian(Eigen::VectorXd (*g)(const Eigen::VectorXd& x, const Eigen::VectorXd& p),
                                   const Eigen::VectorXd& q,
                                   const Eigen::VectorXd& p
                                   ) {
    int n = q.size();
    Eigen::MatrixXd Jg(n,n);
    Eigen::VectorXd dq = Eigen::VectorXd::Zero(n);
    for (int i=0; i<n; i++) {
        dq(i) = NUMERICAL_JACOBIAN_EPS;
        Jg.col(i) = (g(q+dq, p) - g(q, p)) / NUMERICAL_JACOBIAN_EPS;
        dq(i) = 0.0;
    }
    return Jg;
}

Eigen::VectorXd BDF1_step(Eigen::VectorXd (*g)(const Eigen::VectorXd& x, const Eigen::VectorXd& p),
                          const Eigen::VectorXd& xcurr,
                          const Eigen::VectorXd& p,
                          const double h,
                          const int newtonMaxIters = NEWTON_MAX_ITERS,
                          const double newtonTolerance = NEWTON_TOLERANCE
                          ) {

    const int n = xcurr.size();
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n,n);
    Eigen::VectorXd q = xcurr;
    Eigen::VectorXd residual(n);
    Eigen::MatrixXd Jg(n, n);
    int newtonIter;
    for (newtonIter=0; newtonIter<newtonMaxIters; newtonIter++) {
        residual = q - h*g(q,p) - xcurr;
        if (residual.norm() < newtonTolerance)
            break;
        Jg = numerical_jacobian(g, q, p);
        q -= ((I-h*Jg).inverse()) * residual;
    }

    // todo: remove this
    if (newtonIter == newtonMaxIters) {
        std::cout << "WARNING: BDF1_step reached newtonMaxIters\n";
    }

    return q;
}

Eigen::VectorXd BDF2_step(Eigen::VectorXd (*g)(const Eigen::VectorXd& x, const Eigen::VectorXd& p),
                          const Eigen::VectorXd& xcurr,
                          const Eigen::VectorXd& xprev,
                          const Eigen::VectorXd& p,
                          const double h,
                          const double hprev,
                          const int newtonMaxIters = NEWTON_MAX_ITERS,
                          const double newtonTolerance = NEWTON_TOLERANCE
                          ) {

    // initialization
    const int n = xcurr.size();
    Eigen::VectorXd q = xcurr;
    Eigen::VectorXd residual(n);
    Eigen::MatrixXd Jg(n, n);

    const double wn = h/hprev;
    const double a = (1+2*wn)/(1+wn);
    const Eigen::VectorXd b = -(1+wn)*(1+wn)/(1+wn)*xcurr + (wn*wn)/(1+wn)*xprev;
    const Eigen::MatrixXd c = a*Eigen::MatrixXd::Identity(n,n);

    // std::cout << "h = " << h << std::endl;
    // std::cout << "hprev = " << hprev << std::endl;
    // std::cout << "wn = " << wn << std::endl;
    // std::cout << "a = " << a << std::endl;
    // std::cout << "b = " << b.transpose() << std::endl;
    // std::cout << "c = " << c << std::endl;

    // newton's method
    int newtonIter;
    for (newtonIter=0; newtonIter<newtonMaxIters; newtonIter++) {
        residual = a*q - h*g(q,p) + b;
        if (residual.norm() < newtonTolerance)
            break;
        Jg = numerical_jacobian(g, q, p);
        q -= ((c-h*Jg).inverse()) * residual;
    }

    // todo: remove this
    if (newtonIter == newtonMaxIters) {
        std::cout << "WARNING: BDF2_step reached newtonMaxIters\n";
        std::cout << "residual = " << residual.norm() << std::endl;
    }

    // std::cout << "q = " << q.transpose() << std::endl;
    return q;
}

void BDF2_solve(Eigen::VectorXd (*g)(const Eigen::VectorXd& x, const Eigen::VectorXd& p),
                const Eigen::VectorXd& x0,
                const Eigen::VectorXd& p,
                const double t0, const double t_end,
                std::vector<Eigen::VectorXd>& x_output,
                std::vector<double>& t_output,
                const double h0 = 0.01
                ) {
    static const double LTE_NORM_TOLERANCE = 1e-10;
    static const double STEPSIZE_FMAX = 2.414;
    static const double STEPSIZE_FMIN = 0.0;
    static const double STEPSIZE_F = 0.8;
    static const double STEPSIZE_ATOL = 1e-10;
    static const double STEPSIZE_RTOL = 1e-5;

    // init values
    int n = x0.size();
    Eigen::VectorXd x = x0;
    double t = t0;
    double h = h0;
    double hprev = h0;
    Eigen::VectorXd dxdt(n);
    x_output.push_back(x);
    t_output.push_back(t);

    // compute first step with BDF1
    x = BDF1_step(g, x, p, h);
    t += h;
    x_output.push_back(x);
    t_output.push_back(t);

    // compute rest with BDF2
    while (t < t_end) {
        // std::cout << std::endl;
        // std::cout << "t = " << t << std::endl;
        // update values
        x = BDF2_step(g, x, x_output[x_output.size()-2], p, h, hprev);
        t += h;
        hprev = h;

        // update step size
        if (x_output.size() >= 3) {
            // auto LTE = (x/3.0 - x_output[x_output.size()-1] + x_output[x_output.size()-2] - x_output[x_output.size()-3]/3.0);
            double LTE_norm = (x/3.0 - x_output[x_output.size()-1] + x_output[x_output.size()-2] - x_output[x_output.size()-3]/3.0).norm();
            // std::cout << "LTE = " << LTE.transpose() << std::endl;
            // std::cout << "LTE_norm = " << LTE_norm << std::endl;
            // std::cout << "prev x:" << std::endl;
            // std::cout << x.transpose() << std::endl;
            // std::cout << x_output[x_output.size()-1].transpose() << std::endl;
            // std::cout << x_output[x_output.size()-2].transpose() << std::endl;
            // std::cout << x_output[x_output.size()-3].transpose() << std::endl;
            if (LTE_norm < LTE_NORM_TOLERANCE) {
                h = hprev * STEPSIZE_FMAX;
            }
            else {
                double sc = STEPSIZE_ATOL + std::max(x.norm(), (x_output[x_output.size()-1]).norm())*STEPSIZE_RTOL;
                double err = LTE_norm/sc;
                h = hprev * std::min(STEPSIZE_FMAX, std::max(STEPSIZE_FMIN, pow((1.0/err),1.0/3.0)*STEPSIZE_F));
            }
        }

        // save data
        x_output.push_back(x);
        t_output.push_back(t);
    }
}

/*////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                              user provided analytical jacobian
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////*/

Eigen::VectorXd BDF1_step(Eigen::VectorXd (*g)(const Eigen::VectorXd& x, const Eigen::VectorXd& p),
                          Eigen::MatrixXd (*Jg)(const Eigen::VectorXd& x, const Eigen::VectorXd& p),
                          const Eigen::VectorXd& xcurr,
                          const Eigen::VectorXd& p,
                          const double h,
                          const int newtonMaxIters = NEWTON_MAX_ITERS,
                          const double newtonTolerance = NEWTON_TOLERANCE
                          ) {

    const int n = xcurr.size();
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n,n);
    Eigen::VectorXd q = xcurr;
    Eigen::VectorXd residual(n);
    int newtonIter;
    for (newtonIter=0; newtonIter<newtonMaxIters; newtonIter++) {
        residual = q - h*g(q,p) - xcurr;
        if (residual.norm() < newtonTolerance)
            break;
        q -= ((I-h*Jg(q,p)).inverse()) * residual;
    }

    // todo: remove this
    if (newtonIter == newtonMaxIters) {
        std::cout << "WARNING: BDF1_step reached newtonMaxIters\n";
    }

    return q;
}

Eigen::VectorXd BDF2_step(Eigen::VectorXd (*g)(const Eigen::VectorXd& x, const Eigen::VectorXd& p),
                          Eigen::MatrixXd (*Jg)(const Eigen::VectorXd& x, const Eigen::VectorXd& p),
                          const Eigen::VectorXd& xcurr,
                          const Eigen::VectorXd& xprev,
                          const Eigen::VectorXd& p,
                          const double h,
                          const double hprev,
                          const int newtonMaxIters = NEWTON_MAX_ITERS,
                          const double newtonTolerance = NEWTON_TOLERANCE
                          ) {

    // initialization
    const int n = xcurr.size();
    Eigen::VectorXd q = xcurr;
    Eigen::VectorXd residual(n);

    const double wn = h/hprev;
    const double a = (1+2*wn)/(1+wn);
    const Eigen::VectorXd b = -(1+wn)*(1+wn)/(1+wn)*xcurr + (wn*wn)/(1+wn)*xprev;
    const Eigen::MatrixXd c = a*Eigen::MatrixXd::Identity(n,n);

    // newton's method
    int newtonIter;
    for (newtonIter=0; newtonIter<newtonMaxIters; newtonIter++) {
        residual = a*q - h*g(q,p) + b;
        if (residual.norm() < newtonTolerance)
            break;
        q -= ((c-h*Jg(q,p)).inverse()) * residual;
    }

    // todo: remove this
    if (newtonIter == newtonMaxIters) {
        std::cout << "WARNING: BDF2_step reached newtonMaxIters\n";
        std::cout << "residual = " << residual.norm() << std::endl;
    }

    // std::cout << "q = " << q.transpose() << std::endl;
    return q;
}

void BDF2_solve(Eigen::VectorXd (*g)(const Eigen::VectorXd& x, const Eigen::VectorXd& p),
                Eigen::MatrixXd (*Jg)(const Eigen::VectorXd& x, const Eigen::VectorXd& p),
                const Eigen::VectorXd& x0,
                const Eigen::VectorXd& p,
                const double t0, const double t_end,
                std::vector<Eigen::VectorXd>& x_output,
                std::vector<double>& t_output,
                const double h0 = 0.01
                ) {
    static const double LTE_NORM_TOLERANCE = 1e-10;
    static const double STEPSIZE_FMAX = 2.414;
    static const double STEPSIZE_FMIN = 0.0;
    static const double STEPSIZE_F = 0.8;
    static const double STEPSIZE_ATOL = 1e-10;
    static const double STEPSIZE_RTOL = 1e-5;

    // init values
    int n = x0.size();
    Eigen::VectorXd x = x0;
    double t = t0;
    double h = h0;
    double hprev = h0;
    Eigen::VectorXd dxdt(n);
    x_output.push_back(x);
    t_output.push_back(t);

    // compute first step with BDF1
    x = BDF1_step(g, Jg, x, p, h);
    t += h;
    x_output.push_back(x);
    t_output.push_back(t);

    // compute rest with BDF2
    while (t < t_end) {
        // update values
        x = BDF2_step(g, Jg, x, x_output[x_output.size()-2], p, h, hprev);
        t += h;
        hprev = h;

        // update step size
        if (x_output.size() >= 3) {
            double LTE_norm = (x/3.0 - x_output[x_output.size()-1] + x_output[x_output.size()-2] - x_output[x_output.size()-3]/3.0).norm();
            if (LTE_norm < LTE_NORM_TOLERANCE) {
                h = hprev * STEPSIZE_FMAX;
            }
            else {
                double sc = STEPSIZE_ATOL + std::max(x.norm(), (x_output[x_output.size()-1]).norm())*STEPSIZE_RTOL;
                double err = LTE_norm/sc;
                h = hprev * std::min(STEPSIZE_FMAX, std::max(STEPSIZE_FMIN, pow((1.0/err),1.0/3.0)*STEPSIZE_F));
            }
        }

        // save data
        x_output.push_back(x);
        t_output.push_back(t);
    }
}

#ifndef MAIN
// a test ode function
Eigen::VectorXd g(const Eigen::VectorXd& x, const Eigen::VectorXd& p) {
    Eigen::VectorXd dxdt(2);
    dxdt(0) = x(1);
    dxdt(1) = p(0)*(1-x(0)*x(0))*x(1) - x(0);
    // std::cout << "g = " << dxdt.transpose() << std::endl;
    return dxdt;
}
Eigen::MatrixXd Jg(const Eigen::VectorXd& x, const Eigen::VectorXd& p) {
    Eigen::MatrixXd J(2,2);
    J(0,0) = 0;
    J(0,1) = 1;
    J(1,0) = -2*p(0)*x(0)*x(1) - 1;
    J(1,1) = p(0)*(1-x(0)*x(0));
    return J;
}

int main() {
    std::cout << "single thread BDF2 test\n";
    srand(time(NULL));

    Eigen::VectorXd x0(2);
    Eigen::VectorXd p(1);
    p << .1;
    x0 << .2, .3;
    double t0 = 0.0;
    double t_end = 10;
    std::vector<Eigen::VectorXd> x_output;
    std::vector<double> t_output;
    auto timestamp0 = timestamp();
    // BDF2_solve(g, x0, p, t0, t_end, x_output, t_output);
    BDF2_solve(g, Jg, x0, p, t0, t_end, x_output, t_output);
    auto timestamp1 = timestamp();
    std::cout << "BDF2_solve finished in " << time_elapsed_ms(timestamp1, timestamp0) << " ms" << std::endl;

    std::ofstream output_file("output.csv");

    output_file << "t,";
    for (int i=0; i<x_output.size(); i++) {
        output_file << t_output[i];
        if (i < x_output.size()-1) {
            output_file << ",";
        }
    }
    output_file << "\n";

    output_file << "x1,";
    for (int i=0; i<x_output.size(); i++) {
        output_file << x_output[i](0);
        if (i < x_output.size()-1) {
            output_file << ",";
        }
    }
    output_file << "\n";

    output_file << "x2,";
    for (int i=0; i<x_output.size(); i++) {
        output_file << x_output[i](1);
        if (i < x_output.size()-1) {
            output_file << ",";
        }
    }
    output_file << "\n";

    return EXIT_SUCCESS;
}
#endif
