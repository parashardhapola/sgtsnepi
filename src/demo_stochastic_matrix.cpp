/*!
  \file   demo_stochastic_matrix.cpp
  \brief  Demo usage of SG-t-SNE-Pi 
  Demo usage of SG-t-SNE-Pi with a sparse stochastic graph stored in 
  Matrix Market format. The embedding [N-by-d] is exported in a binary 
  file in row-major storage.

  \author Dimitris Floros
  \date   2019-06-21
*/


#include <iostream>
#include <string>
#include <unistd.h>
#include <fstream>

#include "sgtsne.hpp"

int main(int argc, char **argv)
{
  // ~~~~~~~~~~ variable declarations
  int opt;
  tsneparams params;
  std::string filename = "test.mtx";
  std::string out_filename;
  std::string ini_embed;
  coord *y;

  // ~~~~~~~~~~ parse inputs

  // ----- retrieve the (non-option) argument:
  if ( (argc <= 1) || (argv[argc-1] == NULL) || (argv[argc-1][0] == '-') ) {
    // there is NO input...
    std::cerr << "No filename provided!" << std::endl;
    return 1;
  }
  else {
    // there is an input...
    filename = argv[argc-1];
  }

  // ----- retrieve optional arguments

  // Shut GetOpt error messages down (return '?'): 
  opterr = 0;

  while ( (opt = getopt(argc, argv, "l:d:a:m:e:h:p:o:i:")) != -1 ) { 
    switch ( opt ) {
    case 'l':
      sscanf(optarg, "%lf", &params.lambda);
      break;
    case 'd':
      params.d = atoi(optarg);
      break;
    case 'm':
      params.maxIter = atoi(optarg);
      break;
    case 'e':
      params.earlyIter = atoi(optarg);
      break;
    case 'p':
      params.np = atoi(optarg);
      break;
    case 'a':
      sscanf(optarg, "%lf", &params.alpha);
      break;
    case 'h':
      sscanf(optarg, "%lf", &params.h);
      break;
    case 'o':
      out_filename = optarg;
      break;
    case 'i':
      ini_embed = optarg;
      break;
    case '?':  // unknown option...
      std::cerr << "Unknown option: '" << char(optopt) << "'!" << std::endl;
      break;
    }
  }

  if (getWorkers() != params.np && params.np > 0)
    setWorkers( params.np );
  params.np = getWorkers();
  sparse_matrix P = buildPFromMTX( filename.c_str() );
  params.n = P.m;

  double* ini_embed_data = new double[params.n*params.d];
  std::ifstream fin_embed(ini_embed.c_str());
  for (int i = 0; i < params.n*params.d; i++) {
      fin_embed >> ini_embed_data[i];
  }

  y = sgtsne( P, params, ini_embed_data );
  extractEmbeddingText( y, params.n, params.d, out_filename.c_str() );
  free( y );  
}
