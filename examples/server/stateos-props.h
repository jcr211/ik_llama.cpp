#pragma once

// State-OS capability advertised in GET /props as "stateos". Clients (the appliance's "Model state rewind" setting)
// enable model-state rewind only when this object is present with version >= 1: a server without the keyed header
// would load a foreign state instead of refusing it. It is present only when the /slots save/restore routes exist
// (--slot-save-path) and the model identity was computed at startup.

#include "stateos-header.h"

#include <nlohmann/json.hpp>

// null when State-OS save/restore is not usable on this server
inline nlohmann::ordered_json stateos_props_entry(bool slots_enabled, bool identity_ok, bool companion_supported) {
    if (!slots_enabled || !identity_ok) {
        return nullptr;
    }
    return nlohmann::ordered_json{
        { "version",      STATEOS_CONTAINER_VERSION },
        { "keyed_header", true },
        { "companion",    companion_supported }, // this server persists its MTP companion KV in the COMP section
    };
}
