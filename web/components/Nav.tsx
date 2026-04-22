import Link from "next/link";
import styles from "./Nav.module.css";

export function Nav() {
  return (
    <nav className={styles.nav} aria-label="Main">
      <Link href="/" className={styles.brand}>
        Samsung AI
      </Link>
      <div className={styles.links}>
        <Link href="/admin">Admin</Link>
      </div>
    </nav>
  );
}
