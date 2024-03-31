
#include <iostream>
#include <vector>
#include <math.h>
#include <math>
#include <cmath>
#include <algorithm>
using namespace std;
long double pi = 3.14159265358979323846;

vector< vector<long double>> in(int n) {
    vector < vector <long double> > l(n, vector <long double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> l[i][j];
        }
    }
    return l;
}
vector< vector<long double>> eye(int n) {
    vector < vector <long double> > l(n, vector <long double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) l[i][i] = 1;
            else  l[i][j] = 0;
        }
    }
    return l;
}
vector< vector<long double>> zero(int n) {
    vector < vector <long double> > l(n, vector <long double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            l[i][j] = 0;
        }
    }
    return l;
}
vector< vector<long double>> transp(vector < vector <long double> > l) {
    for (int i = 0; i < l.size() - 1; i++) {
        for (int j = i + 1; j < l[0].size(); j++) {
            double x = l[i][j];
            l[i][j] = l[j][i];
            l[j][i] = x;
        }
    }
    return l;
}

double sqrtt(vector <long double> a) {
    long double sqrt_now = 0;
    for (int i = 0; i < a.size(); i++) {
        sqrt_now += a[i] * a[i];
    }
    long double in_sqrt = sqrt_now;
    long double f = sqrt(in_sqrt);
    long double ost = 1;
    long double eps = 0.0000000001;
    while (abs(ost) >= eps) {
        ost = sqrt_now;
        sqrt_now = (sqrt_now + in_sqrt / sqrt_now) / 2;
        ost -= sqrt_now;
    }
    return sqrt_now;

}
double norma1(vector< vector<long double>> a) {
    long double max = 0;
    for (int i = 0; i < a.size(); i++) {
        long double sum = 0;
        for (int j = 0; j < a[0].size(); j++) {
            sum += abs(a[i][j]);
        }
        if (sum > max) max = sum;
    }
    return max;
}
double norma3(vector< vector<long double>> a) {
    long double max = 0;
    for (int i = 0; i < a.size(); i++) {
        long double sum = 0;
        for (int j = 0; j < a[0].size(); j++) {
            sum += abs(a[j][i]);
        }
        if (sum > max) max = sum;
    }
    return max;
}
double normav1(vector<long double> a) {
    long double sum = 0;
    for (int i = 0; i < a.size(); i++) {
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}
bool diag_pr(vector< vector<long double>> a) {
    bool f = 1;
    for (int i = 0; i < a.size(); i++) {
        long double sum = 0;
        for (int j = 0; j < a[0].size(); j++) {
            if (i != j) sum += abs(a[i][j]);
        }
        if (abs(a[i][i]) <= sum) f = 0;
    }
    return f;
}

vector< vector<long double>> multip(vector< vector<long double>> a, vector< vector<long double>> b) {
    vector < vector <long double> > c(a.size(), vector <long double>(b[0].size()));
    for (int i = 0; i < a.size(); i++) {
        for (int k = 0; k < b[0].size(); k++) {
            long double sum_str = 0;
            for (int j = 0; j < a.size(); j++) {
                sum_str += a[i][j] * b[j][k];
            }
            c[i][k] = sum_str;
        }
    }
    return(c);
}
vector<long double> multipv(vector< vector<long double>> a, vector<long double> b) {
    vector <long double> c(a.size());
    for (int i = 0; i < a.size(); i++) {
        long double sum_str = 0;
        for (int j = 0; j < a.size(); j++) {
            sum_str += a[i][j] * b[j];
        }
        c[i] = sum_str;
    }
    return(c);
}
vector< vector<long double>> multiplyvect(vector<long double> a, vector<long double> b) {
    vector < vector <long double> > c(a.size(), vector <long double>(b.size()));
    for (int i = 0; i < a.size(); i++)
        for (int j = 0; j < b.size(); j++)
            c[i][j] = a[i] * b[j];
    return(c);
}

vector< vector<long double>> matarythm(vector< vector<long double>> a, vector< vector<long double>> b, char operation) {
    vector < vector <long double> > c(a.size(), vector <long double>(b[0].size()));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < b[0].size(); j++) {
            if (operation == '+')
                c[i][j] = a[i][j] + b[i][j];
            else if (operation == '-')
                c[i][j] = a[i][j] - b[i][j];
        }
    }
    return(c);
}
vector< vector<long double>> multnum(vector< vector<long double>> a, double k, char operation) {
    vector < vector <long double> > c(a.size(), vector <long double>(a[0].size()));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            if (operation == '*')
                c[i][j] = a[i][j] * k;
            else if (operation == '/')
                c[i][j] = a[i][j] / k;
        }
    }
    return(c);
}


vector<long double> qr_r(int n, vector < vector <long double> > a, vector<long double> b) {
    vector < vector <long double> > ed(n, vector <long double>(n)), helpmat(n, vector <long double>(n));
    vector < vector <long double> > r(n, vector <long double>(n)), q(n, vector <long double>(n)), q_help(n, vector <long double>(n));
    vector<long double> x(n), y(n), z(n), helpv(n), w(n);
    bool dia = 1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) ed[i][j] = 1;
            else ed[i][j] = 0;
            if (i > j) {
                if (a[i][j] != 0) dia = 0;
            }
        }
    }
    q = ed;
    int k = 0;
    r = a;
    if (dia == 0) {
        while (k < n - 1) {
            vector < vector <long double> > edk(n - k, vector <long double>(n - k)), helpmatk(n - k, vector <long double>(n - k));
            vector < vector <long double> > rk(n - k, vector <long double>(n - k)), qk(n - k, vector <long double>(n - k));
            vector<long double> yk(n - k), zk(n - k), helpvk(n - k), wk(n - k);
            int ll = k;
            for (int i = 0; i < n - k; i++) {
                int dd = k;
                for (int j = 0; j < n - k; j++) {
                    if (i == j) edk[i][j] = 1;
                    else edk[i][j] = 0;
                    rk[i][j] = r[ll][dd];
                    dd++;
                }
                yk[i] = r[ll][k];
                zk[i] = 0;
                ll++;
            }
            qk = edk;
            double alpha = normav1(yk);
            zk[0] = alpha;
            helpvk = yk; helpvk[0] -= zk[0];
            double ro = normav1(helpvk);
            for (int i = 0; i < n - k; i++) {
                wk[i] = helpvk[i] / ro;
            }
            helpmatk = multnum(multiplyvect(wk, wk), 2, '*');
            qk = matarythm(edk, helpmatk, '-');
            rk = multip(qk, rk);
            int l = 0;
            q_help = ed;
            for (int i = k; i < n; i++) {
                int d = 0;
                for (int j = k; j < n; j++) {
                    r[i][j] = rk[l][d];
                    q_help[i][j] = qk[l][d];
                    d++;
                }
                l++;
            }
            k++;
            q = multip(q, q_help);
        }

    }
    if (dia == 0) y = multipv(transp(q), b);
    else y = b;

    x[n - 1] = y[n - 1] / r[n - 1][n - 1];
    for (int i = n - 2; i >= 0; i--) {
        long double sum_x = 0;
        for (int k = i + 1; k < n; k++) {
            sum_x += r[i][k] * x[k];
        }
        x[i] = (y[i] - sum_x) / r[i][i];
    }
    return(x);
}

vector<long double> gauss(int n, vector < vector <long double> > a, vector<long double> y)
{
    vector<long double> x(n);
    long double max;
    int k, index;
    const double eps = 0.00001;  // точность
    k = 0;
    while (k < n)
    {
        // Поиск строки с максимальным a[i][k]
        max = abs(a[k][k]);
        index = k;
        for (int i = k + 1; i < n; i++)
        {
            if (abs(a[i][k]) > max)
            {
                max = abs(a[i][k]);
                index = i;
            }
        }
        // Перестановка строк
        if (max < eps)
        {
            // нет ненулевых диагональных элементов
            std::cout << "Решение получить невозможно из-за нулевого столбца ";
            std::cout << index << " матрицы A" << endl;

        }
        for (int j = 0; j < n; j++)
        {
            long double temp = a[k][j];
            a[k][j] = a[index][j];
            a[index][j] = temp;
        }
        long double temp = y[k];
        y[k] = y[index];
        y[index] = temp;
        // Нормализация уравнений
        for (int i = k; i < n; i++)
        {
            long double temp = a[i][k];
            if (abs(temp) < eps) continue; // для нулевого коэффициента пропустить
            for (int j = 0; j < n; j++)
                a[i][j] = a[i][j] / temp;
            y[i] = y[i] / temp;
            if (i == k)  continue; // уравнение не вычитать само из себя
            for (int j = 0; j < n; j++)
                a[i][j] = a[i][j] - a[k][j];
            y[i] = y[i] - y[k];
        }
        k++;
    }
    // обратная подстановка
    for (k = n - 1; k >= 0; k--)
    {
        x[k] = y[k];
        for (int i = 0; i < k; i++)
            y[i] = y[i] - a[i][k] * x[k];
    }
    return x;
}
vector < vector <long double> > lagr_coef(int n, vector<long double> xi) {
    vector < vector <long double> > ll(n, vector<long double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ll[i][j] = pow(xi[i], j);
        }
    }
    return ll;
}

void PMetod() {
    int n;
    cin >> n;
    vector < vector <long double> > A(n, vector<long double>(n));
    A = in(n);
    vector <long double> y(n), lambda(n), lambda_k(n), z(n);
    for (int i = 0; i < n; i++) {
        y[i] = 0; lambda[i] = 0;
        z[i] = 0;
    }
    y[0] = 1; z[0] = 1;
    long double delta = 1, eps = 0.001, I = 0, eigenvalue = 0, sumlam = 0, norm = 1, rtol = 0.000001;
    while (delta > eps) {
        delta = 0;
        y = multipv(A, z);
        norm = normav1(y);
        I = 0; sumlam = 0;
        for (int i = 0; i < n; i++) {
            if (abs(z[i]) > 0.00000001) {
                lambda_k[i] = lambda[i];
                lambda[i] = y[i] / z[i];
                delta += (lambda_k[i] - lambda[i]) * (lambda_k[i] - lambda[i]);
                I++;
                sumlam += lambda[i];
            }

            z[i] = y[i] / norm;
        }

        delta = sqrt(delta);
        eps = rtol * max(normav1(lambda_k), normav1(lambda));

    }
    eigenvalue = sumlam / I;

    cout << "\neigenvalue = " << eigenvalue;
    cout << "\nvect: ";
    for (int i = 0; i < n; i++)
        cout << z[i] << ' ';

}

void RevPM() {
    int n;
    cin >> n;
    /*
1 2 3 4
4 5 6 7
6 7 8 9
7 8 9 10
    */

    vector < vector <long double> > A(n, vector<long double>(n)), M(n, vector<long double>(n));
    vector < vector <long double> > E(n, vector<long double>(n)),
    A = in(n);
    E = eye(n);
    
    vector <long double> y(n), lambda(n), lambda_k(n), z(n), mu(n);
    long double sigma = 0.04, norm = 1;
    for (int i = 0; i < n; i++) {
        y[i] = 0; lambda[i] = 0;
        z[i] = 0;
        mu[i] = 0;
    }
    y = { 1, -2, 1, 0 };
    norm = normav1(y);
    for (int i = 0; i < n; i++) {
        z[i] = y[i] / norm;
    }



    long double delta = 1, eps = 0.001, I = 0, eigenvalue = 0, sumlam = 0,  rtol = 0.000001;
    while (delta > eps) {
        delta = 0;
        M = multnum(E, sigma, '*');
        M = matarythm(A, M, '-');
        y = gauss(n, M, z);
        norm = normav1(y);
        for (int i = 0; i < n; i++) {
            
        }
        I = 0; sumlam = 0;
        for (int i = 0; i < n; i++) {
            if (abs(z[i]) > 0.00000001) {
                mu[i] = z[i] / y[i];
                lambda_k[i] = lambda[i];
                lambda[i] = y[i] / z[i];
                
                I++;
                sumlam += mu[i];
            }

            z[i] = y[i] / norm;
            delta += (sigma - z[i]) * (sigma - z[i]);
        }

        delta = sqrt(delta);
        eps = rtol * max(normav1(lambda_k), normav1(lambda));

    }
    eigenvalue = sumlam / I;

    cout << "\neigenvalue = " << eigenvalue;
    cout << "\nvect: ";
    for (int i = 0; i < n; i++)
        cout << z[i] << ' ';
}

void QR() {
    int n;
    cin >> n;
    /*
1 2 3 4
4 5 6 7
6 7 8 9
7 8 9 10
    */

    vector < vector <long double> > A(n, vector<long double>(n)), H(n, vector<long double>(n));
    vector < vector <long double> > E(n, vector<long double>(n)), B(n, vector<long double>(n));
    A = in(n);
    E = eye(n);

    H = A;
    B = A;
    vector <long double>  v(n);
    long double sign, s, m;

    // приведем матрицу к форме Хессенберга:
    for (int t = 0; t < n - 1; t++) {
        sign = 0, s = 0, m = 0;
        sign = B[1 + t][t] / abs(B[1 + t][t]);
        for (int i = t + 1; i < n; i++) {
            s += B[i][t] * B[i][t];
        }
        s = sign * sqrt(s);
        m = 1 / (2 * s * (s - B[1 + t][t]));

        for (int i = 0; i < t + 2; i++) {
            v[i] = 0;
        }
        v[1 + t] = m * (H[1 + t][t] - s);
        for (int i = 2 + t; i < n; i++) {
            v[i] = m * B[i][t];
        }
        H = matarythm(E, multnum(multiplyvect(v, v), 2, '*'), '-');
        B = multip(multip(H, B), H);

    }
    vector <long double> y(n), lambda(n), lambda_k(n), z(n), mu(n),
    
    long double delta = 1, eps = 0.001, I = 0, eigenvalue = 0, sumlam = 0, rtol = 0.000001;
    while (delta > eps) {
        delta = 0;
        M = multnum(E, sigma, '*');
        M = matarythm(A, M, '-');
        y = gauss(n, M, z);
        norm = normav1(y);
        for (int i = 0; i < n; i++) {

        }
        I = 0; sumlam = 0;
        for (int i = 0; i < n; i++) {
            if (abs(z[i]) > 0.00000001) {
                mu[i] = z[i] / y[i];
                lambda_k[i] = lambda[i];
                lambda[i] = y[i] / z[i];

                I++;
                sumlam += mu[i];
            }

            z[i] = y[i] / norm;
            delta += (sigma - z[i]) * (sigma - z[i]);
        }

        delta = sqrt(delta);
        eps = rtol * max(normav1(lambda_k), normav1(lambda));

    }
    eigenvalue = sumlam / I;

    cout << "\neigenvalue = " << eigenvalue;
    cout << "\nvect: ";
    for (int i = 0; i < n; i++)
        cout << z[i] << ' ';
}
int main() {
    cout << "\nPM\n";
    PMetod();
    cout << "\nRevPM\n";
    RevPM();
}


/*
    long double delta = 1, eps = 0.001, I = 0, eigenvalue = 0, sumlam = 0, rtol = 0.000001;
    while (delta > eps) {
        delta = 0;
        M = multnum(E, sigma, '*');
        M = matarythm(A, M, '-');
        y = gauss(n, M, z);
        norm = normav1(y);
        for (int i = 0; i < n; i++) {

        }
        I = 0; sumlam = 0;
        for (int i = 0; i < n; i++) {
            if (abs(z[i]) > 0.00000001) {
                mu[i] = z[i] / y[i];
                lambda_k[i] = lambda[i];
                lambda[i] = y[i] / z[i];

                I++;
                sumlam += mu[i];
            }

            z[i] = y[i] / norm;
            delta += (sigma - z[i]) * (sigma - z[i]);
        }

        delta = sqrt(delta);
        eps = rtol * max(normav1(lambda_k), normav1(lambda));

    }
    eigenvalue = sumlam / I;

    cout << "\neigenvalue = " << eigenvalue;
    cout << "\nvect: ";
    for (int i = 0; i < n; i++)
        cout << z[i] << ' ';*/