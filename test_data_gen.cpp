#include <bits/stdc++.h>
using namespace std;

int main()
{
    srand(time(NULL));
    std::ofstream X_train("X.txt");
    std::ofstream y_train("y.txt");
    if (!X_train.is_open() || !y_train.is_open())
    {
        cout << "File not opened\n";
        return 0;
    }
    
    for (int i = 0; i < 11; i++)
    {
        for (int j = 0; j < 11; j++)
        {
            int gen_or_not_para = rand() % 3;
            if (gen_or_not_para < 1)
            {
                X_train << i << ' ' << j << '\n';
                if (i + j < 10)
                {
                    y_train << '1' << '\n';
                }
                else
                {
                    y_train << '0' << '\n';
                }
            }
        }
    }
    
    return 0;
}