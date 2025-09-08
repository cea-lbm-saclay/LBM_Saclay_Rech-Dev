/*
*/


#include "Models_GP_Mixt_Ternary.h"

namespace PBM_GP_MIXT_TERNARY
{

// ================================================
//
// Models for mixt formulation of grand potential problems
//
// ================================================


using Tag2Quadra = ModelParams::Tag2Quadra;

void
ModelParams::showParams()
{
    std::cout << "W :    " << W << std::endl;
    std::cout << "mailles par W :    " << W / dx << std::endl;
    std::cout << "W/lambda :    " << W / lambda << std::endl;
    std::cout << "lambda*Mphi :    " << Mphi * lambda << std::endl;
    std::cout << "counter_term :    " << counter_term << std::endl;
    std::cout << "at_current :    " << at_current << std::endl;
    std::cout << "init :" << initType << "." << std::endl;
    std::cout << "initClA :" << initClA << "." << std::endl;
    std::cout << "initCsA :" << initCsA << "." << std::endl;
    std::cout << "initClB :" << initClB << "." << std::endl;
    std::cout << "initCsB :" << initCsB << "." << std::endl;
    std::cout << "resolution of diffusion length (root(min(D)*dt)/dx):  " << SQRT(FMIN(FMIN(DA1, DA0), FMIN(DB1, DB0)) * dt) / dx << std::endl;
    std::cout << "diffusion length to interface ratio (W**2/(min(D)*dt)):  " << W / SQRT(FMIN(FMIN(DA1, DA0), FMIN(DB1, DB0)) * dt) << "   (this is also how many time steps before the profile gets out of the interface)" << std::endl;
    //~ real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
    std::cout << "tauPhi :" << (0.5 + (e2 * Mphi * dt / SQR(dx))) << std::endl;
    std::cout << "quadra case:" << std::endl;
    std::cout << "tauCAmin :" << (0.5 + (e2 / gammaA * FMIN(DA1, DA0) * dt / SQR(dx))) << std::endl;
    std::cout << "tauCAmax :" << (0.5 + (e2 / gammaA * FMAX(DA1, DA0) * dt / SQR(dx))) << std::endl;
    std::cout << "tauCBmin :" << (0.5 + (e2 / gammaB * FMIN(DB1, DB0) * dt / SQR(dx))) << std::endl;
    std::cout << "tauCBmax :" << (0.5 + (e2 / gammaB * FMAX(DB1, DB0) * dt / SQR(dx))) << std::endl;
    std::cout << "quadra with solid case:" << std::endl;
    std::cout << "tauCA :" << (0.5 + (e2 / gammaA * dt / SQR(dx))) << std::endl;
    std::cout << "tauCB :" << (0.5 + (e2 / gammaB * dt / SQR(dx))) << std::endl;
    std::cout << "dilute case:" << std::endl;
    std::cout << "tauClA :" << (0.5 + (e2 / gammaA * initClA * DA1 * dt / SQR(dx))) << std::endl;
    std::cout << "tauCsA :" << (0.5 + (e2 / gammaA * initCsA * DA0 * dt / SQR(dx))) << std::endl;
    std::cout << "tauClB :" << (0.5 + (e2 / gammaB * initClB * DB1 * dt / SQR(dx))) << std::endl;
    std::cout << "tauCsB :" << (0.5 + (e2 / gammaB * initCsB * DB0 * dt / SQR(dx))) << std::endl;
    // std::cout << "dilute case with instant diff:" << std::endl;
    // Tag2DiluteInstantDiffusion tagID;
    // real_t mu, c, tau;
    // mu = compute_muA(tagID, 0, initCsA);
    // c = compute_csA(tagID, mu);
    // tau = 0.5 + (e2 * (c * DA0) * dt / SQR(dx) / gammaA);
    // std::cout << "tauCsA :" << tau << std::endl;

    // mu = compute_muA(tagID, 1, initClA);
    // c = compute_csA(tagID, mu);
    // tau = 0.5 + (e2 * (c * DA0) * dt / SQR(dx) / gammaA);
    // std::cout << "tauClA :" << tau << std::endl;

    // mu = compute_muB(tagID, 0, initCsB);
    // c = compute_csB(tagID, mu);
    // tau = 0.5 + (e2 * (c * DB0) * dt / SQR(dx) / gammaB);
    // std::cout << "tauCsB :" << tau << std::endl;

    // mu = compute_muB(tagID, 1, initClB);
    // c = compute_csB(tagID, mu);
    // tau = 0.5 + (e2 * (c * DB0) * dt / SQR(dx) / gammaB);
    // std::cout << "tauClB :" << tau << std::endl;
};

ModelParams::ModelParams(const ConfigMap& configMap,const LBMParams& params)
{

    dt = params.dt;
    dx = params.dx;
    time = 0;
    e2 = configMap.getFloat("lbm", "e2", 3.0);
    W = configMap.getFloat("params", "W", 0.005);
    Mphi = configMap.getFloat("params", "Mphi", 1.2);
    lambda = configMap.getFloat("params", "lambda", 230.0);

    epsiL = configMap.getFloat("params", "epsiL", 1.0);
    epsiS = configMap.getFloat("params", "epsiS", 1.0);
    mlA = configMap.getFloat("params", "mlA", 0.1);
    mlB = configMap.getFloat("params", "mlB", 0.1);
    msA = configMap.getFloat("params", "msA", 0.2);
    msB = configMap.getFloat("params", "msB", 0.2);
    Q = configMap.getFloat("params", "Q", -0.04);

    DA1 = configMap.getFloat("params", "DA1", 1.0);
    DA0 = configMap.getFloat("params", "DA0", 1.0);
    DB1 = configMap.getFloat("params", "DB1", 1.0);
    DB0 = configMap.getFloat("params", "DB0", 1.0);

    gammaA = configMap.getFloat("params", "gammaA", 1.0);
    gammaB = configMap.getFloat("params", "gammaB", 1.0);

    counter_term = configMap.getFloat("params", "counter_term", 0.0);
    at_current = configMap.getFloat("params", "at_current", 0.0);
    diffusion_current = configMap.getFloat("params", "diffusion_current", 1.0);
    diffusion_current2 = configMap.getFloat("params", "diffusion_current2", 0.0);

    elA = configMap.getFloat("params", "elA", 0.1);
    esA = configMap.getFloat("params", "esA", 0.1);
    elB = configMap.getFloat("params", "elB", 0.1);
    esB = configMap.getFloat("params", "esB", 0.1);
    elC = configMap.getFloat("params", "elC", 0.1);
    esC = configMap.getFloat("params", "esC", 0.1);

    x0 = configMap.getFloat("init", "x0", 0.0);
    y0 = configMap.getFloat("init", "y0", 0.0);
    z0 = configMap.getFloat("init", "z0", 0.0);
    r0 = configMap.getFloat("init", "r0", 0.2);

    x2 = configMap.getFloat("init", "x2", 0.0);
    y2 = configMap.getFloat("init", "y2", 0.0);
    z2 = configMap.getFloat("init", "z2", 0.0);
    r2 = configMap.getFloat("init", "r2", 0.2);

    initClA = configMap.getFloat("init", "initClA", 0.1);
    initCsA = configMap.getFloat("init", "initCsA", 0.2);
    initClB = configMap.getFloat("init", "initClB", 0.1);
    initCsB = configMap.getFloat("init", "initCsB", 0.2);

    sign = configMap.getFloat("init", "sign", 1);
    t0 = configMap.getFloat("init", "t0", 1.0);

    freq = configMap.getFloat("init", "freq", 5);

    A1A = configMap.getFloat("init", "A1A", 0);
    B1A = configMap.getFloat("init", "B1A", 0);
    A0A = configMap.getFloat("init", "A0A", 0);
    B0A = configMap.getFloat("init", "B0A", 0);
    A1B = configMap.getFloat("init", "A1B", 0);
    B1B = configMap.getFloat("init", "B1B", 0);
    A0B = configMap.getFloat("init", "A0B", 0);
    B0B = configMap.getFloat("init", "B0B", 0);
    xi = configMap.getFloat("init", "xi", 0);
    init_time = configMap.getFloat("init", "init_time", 0);

    initCsAA = configMap.getFloat("init", "initCsAA", 0.2);
    initCsAB = configMap.getFloat("init", "initCsAB", 0.2);
    initCsBA = configMap.getFloat("init", "initCsBA", 0.2);
    initCsBB = configMap.getFloat("init", "initCsBB", 0.2);

    use_sharp_init = configMap.getBool("init", "sharp_init", false);

    read_data_phi = configMap.getBool("init", "read_data_phi", false);
    file_data_phi = configMap.getString("init", "file_data_phi", "unknown");

    initType = PHASE_FIELD_INIT_UNDEFINED;

    std::string initTypeStr = std::string(configMap.getString("init", "init_type", "unknown"));

    if (initTypeStr == "vertical")
    {
        initType = PHASE_FIELD_INIT_VERTICAL;
    }
    else if (initTypeStr == "sphere")
    {
        initType = PHASE_FIELD_INIT_SPHERE;
    }
    else if (initTypeStr == "2sphere")
    {
        initType = PHASE_FIELD_INIT_2SPHERE;
    }
    else if (initTypeStr == "square")
    {
        initType = PHASE_FIELD_INIT_SQUARE;
    }
    else if (initTypeStr == "data")
    {
        initType = PHASE_FIELD_INIT_DATA;
    }
    else if (initTypeStr == "ternary_random")
    {
        initType = PHASE_FIELD_RANDOM_TERNARY_GLASS;
    }
    else if (initTypeStr == "ternary_random_sphere")
    {
        initType = PHASE_FIELD_RANDOM_TERNARY_GLASS_SPHERE;
    }
    else if (initTypeStr == "vertical_perturbed")
    {
        initType = PHASE_FIELD_INIT_VERTICAL_PERTURBED;
    }
    else if (initTypeStr == "vertical_perturbed_random")
    {
        initType = PHASE_FIELD_INIT_VERTICAL_PERTURBED_RANDOM;
    }
    else if (initTypeStr == "erfc_1int")
    {
        initType = PHASE_FIELD_INIT_VERTICAL_ERFC_1INT;
    }
    else if (initTypeStr == "erfc_3int")
    {
        initType = PHASE_FIELD_INIT_VERTICAL_ERFC_3INT;
    }
    else if (initTypeStr == "perco")
    {
        initType = PHASE_FIELD_INIT_TERNARY_PERCO;
    }

    use_connected_components = configMap.getBool("ccparams", "use_connected_components", false);
    print_cc_trace = configMap.getBool("ccparams", "print_cc_trace", false);

    CC_phi_threshold = configMap.getFloat("ccparams", "CC_phi_threshold", 0.5);

    apply_virtual_volume = configMap.getBool("ccparams", "apply_virtual_volume", false);
    virtual_volume = configMap.getFloat("ccparams", "virtual_volume", 0);
    
    virtual_volume_anchor_x = configMap.getInteger("ccparams", "virtual_volume_anchor_x", 0);
    virtual_volume_anchor_y = configMap.getInteger("ccparams", "virtual_volume_anchor_y", 0);
    virtual_volume_anchor_z = configMap.getInteger("ccparams", "virtual_volume_anchor_z", 0);
    
    // std::string side = configMap.getString("ccparams", "virtual_volume_boundary", "xmin");
    // auto map_face_to_id = maps::getMAP_FACE_NAME_TO_ID();
    // if (map_face_to_id.find(side) != map_face_to_id.end())
    // {
        // virtual_volume_application_bdy = map_face_to_id[side];
    // }
    // else
    // {
        // virtual_volume_application_bdy = FACE_XMIN;
    // }

    cclabel_connected_to_virtual_volume = 1;
    cA_in_virtual_volume = configMap.getFloat("init", "initClA", 0.1) * virtual_volume;
    cB_in_virtual_volume = configMap.getFloat("init", "initClB", 0.1) * virtual_volume;
    
    if (params.myRank == 0)
    {
        showParams();
    }
}

} // namespace
