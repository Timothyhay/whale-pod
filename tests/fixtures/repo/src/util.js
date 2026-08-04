export function formatUser(name, age) {
  return `${name}:${age}`;
}
export class Greeter {
  constructor(prefix) { this.prefix = prefix; }
  greet(who) { return this.prefix + who; }
}
export interface Shape { area(): number }
