#pragma once

// State-OS capability advertised in GET /props as "stateos". Clients (the appliance's "Model state rewind" setting)
// enable model-state rewind only when this object is present with version >= 1: a server without the keyed header
// would load a foreign state instead of refusing it.

#include "stateos-header.h"

#include <nlohmann/json.hpp>

inline nlohmann::ordered_json stateos_props_capability(bool companion_supported) {
    return nlohmann::ordered_json{
        { "version",      STATEOS_CONTAINER_VERSION },
        { "keyed_header", true },
        { "companion",    companion_supported }, // this server persists its MTP companion KV in the COMP section
    };
}
