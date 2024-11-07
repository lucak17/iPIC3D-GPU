
#include "cudaTypeDef.cuh"
#include "velocityHistogram.cuh"
#include "particleArraySoACUDA.cuh"

namespace velocityHistogram
{

using namespace particleArraySoA;

/**
 * @brief kernel function to calculate the velocity histogram for one species
 * @details the 3 histograms have been updated in advance
 * 
 * @param pclArray the particle array
 * @param histogramCUDAPtrUV the histogram for UV
 * @param histogramCUDAPtrVW the histogram for VW
 * @param histogramCUDAPtrUW the histogram for UW
 */
__global__ void velocityHistogramKernel(velocitySoA* pclArraySoA, velocityHistogramCUDA* histogramCUDAPtrUV, 
                                                    velocityHistogramCUDA* histogramCUDAPtrVW, 
                                                    velocityHistogramCUDA* histogramCUDAPtrUW){

    int pidx = threadIdx.x + blockIdx.x * blockDim.x;
    if(pidx >= pclArraySoA->getNOP())return;

    const cudaCommonType uvw[3] = {pclArraySoA->getElement(particleArraySoA::particleArraySoAElement::U)[pidx], 
                                    pclArraySoA->getElement(particleArraySoA::particleArraySoAElement::V)[pidx], 
                                    pclArraySoA->getElement(particleArraySoA::particleArraySoAElement::W)[pidx]};
    const cudaCommonType uv[2] = {uvw[0], uvw[1]};
    const cudaCommonType vw[2] = {uvw[1], uvw[2]};
    const cudaCommonType uw[2] = {uvw[0], uvw[2]};

    histogramCUDAPtrUV->addData(uv, 1);
    histogramCUDAPtrVW->addData(vw, 1);
    histogramCUDAPtrUW->addData(uw, 1);


}



} // namespace velocityHistogram







