from __future__ import annotations

from chan.validate import check_bi_direction, check_bi_time_order, check_center_range


def main() -> None:
    codes = ["SH.688122", "SZ.000426", "SZ.300624"]
    freq = "day"
    begin = "2023-01-01"
    end = "2023-12-31"

    for code in codes:
        print(f"=== {code} {freq} ===")
        d_err = check_bi_direction(code, freq, begin, end)
        t_err = check_bi_time_order(code, freq)
        c_err = check_center_range(code, freq)
        print(f"direction_errors: {d_err}")
        print(f"time_errors: {t_err}")
        print(f"center_errors: {c_err}")


if __name__ == "__main__":
    main()