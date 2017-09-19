#include <math.h>
#include <string>
#include "loss.h"

using namespace std;

namespace Losses
{
	const Loss squaredError(
		"squaredError",
		[](double truth, double prediction) { 
			double tmp = prediction-truth;
			return tmp*tmp; 
		},
		[](double truth, double prediction) { return 2.0*(prediction-truth); }
	);

	const Loss absoluteError(
		"absoluteError",
		[](double truth, double prediction) { 
			double tmp = truth-prediction;
			return (tmp>0.0) ? tmp : -tmp; 
		},
		[](double truth, double prediction) { 
			double tmp = truth-prediction;
			return (tmp>0.0) ? 1.0 : -1.0; 
		}
	);

	const Loss binaryEntropy(
		"binaryEntropy",
		[](double truth, double prediction) { 
			prediction = (prediction < 1e-8) ? 1e-8: prediction;
			prediction = (prediction > 1 - 1e-8) ? 1 - 1e-8: prediction;
			return -truth*log(prediction) - (1-truth)*log(1-prediction); 
		},
		[](double truth, double prediction) { 
			prediction = (prediction < 1e-8) ? 1e-8: prediction;
			prediction = (prediction > 1 - 1e-8) ? 1 - 1e-8: prediction;
			return -truth/prediction + (1-truth)/(1-prediction);
		}
	);
}

