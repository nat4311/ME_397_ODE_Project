#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>

struct DataLine {
    std::string name;
    std::vector<double> data;

    DataLine(std::string line) {
        std::stringstream ss(line);
        std::string token;
        std::getline(ss, token, ',');
        name = token;
        while (std::getline(ss, token, ',')) {
            data.push_back(stod(token));
        }
    }

    void print() {
        std::cout << name << " = [";
        for (int i=0; i<data.size(); i++) {
            std::cout << data[i];
            if (i!=data.size()-1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
};

