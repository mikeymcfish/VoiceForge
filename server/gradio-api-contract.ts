type UnknownRecord = Record<string, unknown>;

function isRecord(value: unknown): value is UnknownRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isSafeInternalState(parameter: UnknownRecord): boolean {
  return (
    parameter.parameter_name === null &&
    parameter.component === "state" &&
    parameter.hidden === true &&
    parameter.parameter_has_default === true &&
    parameter.parameter_default === null
  );
}

/**
 * Gradio's client exposes server-owned `gr.State` inputs alongside public API
 * parameters. They are not caller-controlled, but are represented with a null
 * parameter name. Ignore only the exact, non-callable state shape that Gradio
 * currently returns; every other unnamed or hidden input remains unsafe.
 */
export function publicGradioParameterNames(parameters: unknown): string[] {
  if (!Array.isArray(parameters)) {
    throw new Error("The endpoint parameter list is missing or malformed.");
  }

  const names: string[] = [];
  parameters.forEach((parameter, index) => {
    if (!isRecord(parameter)) {
      throw new Error(`Parameter ${index + 1} is malformed.`);
    }

    if (typeof parameter.parameter_name === "string" && parameter.parameter_name.length > 0) {
      if (parameter.hidden === true) {
        throw new Error(`Public parameter ${parameter.parameter_name} is unexpectedly hidden.`);
      }
      names.push(parameter.parameter_name);
      return;
    }

    if (!isSafeInternalState(parameter)) {
      throw new Error(`Parameter ${index + 1} is unnamed and is not a safe internal Gradio state.`);
    }
  });

  return names;
}

export function endpointParameters(api: unknown, endpointName: string): string[] | undefined {
  if (!isRecord(api) || !isRecord(api.named_endpoints)) return undefined;
  const endpoint = api.named_endpoints[endpointName];
  if (!isRecord(endpoint)) return undefined;
  return publicGradioParameterNames(endpoint.parameters);
}

export function exactParameterNames(
  actual: readonly string[] | undefined,
  expected: readonly string[]
): boolean {
  return Boolean(
    actual &&
    actual.length === expected.length &&
    actual.every((name, index) => name === expected[index])
  );
}
